#!/usr/bin/env python3
# sam2_segment.py
# Author: <your name>
# -----------------------------------------------------------------------------
"""
利用 Florence-2 生成目标人框 + SAM-2 进行跨帧视频分割。

必选参数
---------
--video-dir          视频帧所在文件夹（必须已按 000001.jpg 连续命名）

可选参数
---------
--output-dir         分割掩膜输出目录（默认同级 masks）
--model-id           Florence-2 Hugging Face 权重 (default: microsoft/Florence-2-large)
--sam2-id            SAM-2 Hugging Face 权重 (default: facebook/sam2-hiera-large)
--text               自定义描述文本，用于生成人框
--ann-frame-idx      与 SAM-2 交互的关键帧索引 (default: 0)
--ann-obj-id         SAM-2 里分配的对象 ID (default: 4)
--device             推理设备 (cuda / cpu) (default: cuda)
"""
# -----------------------------------------------------------------------------
from __future__ import annotations
from email.mime import image
import argparse, os, sys, pathlib, json
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.sam2_video_predictor import SAM2VideoPredictor
from pathlib import Path
from typing import List, Union, Literal
from alive_progress import alive_bar

# ------------------------------Florence2 包装---------------------------------
def get_person_bbox(image: Image.Image,
                    task: str,
                    prompt: str,
                    model_id: str = "microsoft/Florence-2-large",
                    device: str = "cuda") -> np.ndarray:
    """使用 Florence-2 caption-to-phrase-grounding 预测“person”框"""
    task_prompt = task
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = (AutoModelForCausalLM
             .from_pretrained(model_id, trust_remote_code=True,
                              torch_dtype="auto")
             .eval().to(device))

    inputs = processor(
        text=task_prompt + prompt,
        images=image,
        return_tensors="pt").to(device, torch.float16)

    with torch.inference_mode():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False, num_beams=3)
    decoded = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        decoded, task=task_prompt, image_size=(image.width, image.height))

    labels = parsed[task_prompt]["labels"]
    for i, lab in enumerate(labels):
        if lab.lower().startswith("the person") or lab.lower().endswith("person"):
            print(f"parsed[task_prompt][labels][{i}]: {lab}, parsed[task_prompt][bboxes][{i}]:{parsed[task_prompt]['bboxes'][i]}")
            return np.array(parsed[task_prompt]["bboxes"][i], dtype=np.float32)

    raise RuntimeError("未在描述中检测到 person 框")

# ------------------------------文本处理---------------------------------------
def read_txt(path: Union[str, Path],
             mode: Literal["text", "lines"] = "text",
             encoding: str = "utf-8") -> Union[str, List[str]]:
    """
    读取 txt 文件

    参数
    ----
    path      : 文件路径
    mode      : "text" 返回无换行符的完整字符串；
                "lines" 返回去除换行符的行列表
    encoding  : 文件编码，默认 utf-8
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    with path.open("r", encoding=encoding) as f:
        if mode == "text":
            # 一次性读取后删除所有换行符并返回
            return f.read().replace("\n", "").replace("\r", "")
        elif mode == "lines":
            # 逐行读取并去掉每行末尾的换行符
            return [line.rstrip("\n\r") for line in f]
        else:
            raise ValueError('mode 只能取 "text" 或 "lines"')

# ------------------------------主流程-----------------------------------------
def run(args: argparse.Namespace) -> None:
    root_dir = pathlib.Path(args.video)
    image_dir = root_dir / "images"
    sub_image_dir = [p.name for p in image_dir.iterdir() if p.is_dir()]
    print(f'Processing {len(sub_image_dir)} 个子目录: {sub_image_dir}')
    if not image_dir.is_dir():
        sys.exit(f"[ERROR] video_dir 不存在: {image_dir}")

    if args.text is None:
        text_path = root_dir / "motion_text.txt"
        text = read_txt(text_path)
        print(text)

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    # 读取首帧 → 生成人框
    for sub_dir in sub_image_dir:
        print(f"[INFO] 处理子目录: {sub_dir}")
        rgb_path = image_dir / sub_dir /"ori_imgs"

        first_img = Image.open(sorted(rgb_path.glob("*.jpg"))[0])
        bbox = get_person_bbox(
            first_img, args.task, text)

        # 初始化 SAM-2
        
        inference_state = predictor.init_state(video_path=str(rgb_path))

        # 与指定关键帧交互
        _, out_obj_ids, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=args.ann_frame_idx,
            obj_id=args.ann_obj_id,
            box=bbox)

        # Propagate
        masks_per_frame: dict[int, dict[int, np.ndarray]] = {}
        for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            masks_per_frame[f_idx] = {
                obj_id: (mask_logits[i] > 0).cpu().numpy().astype(np.uint8) * 255
                for i, obj_id in enumerate(obj_ids)
            }

        # 保存
        if args.output is None:
            out_dir = image_dir / sub_dir / "masks"
        else:
            out_dir = pathlib.Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_names = sorted(rgb_path.glob("*.jpg"))
        with alive_bar(len(frame_names),
               title="Saving masks",
               length=80,
               max_cols=120,
               spinner="loving", # 这是一个常见的 spinner
               bar="halloween"    # 这是一个常见的 bar
              ) as bar:
            for i, fname in enumerate(frame_names):
                for obj_id, mask in masks_per_frame[i].items():
                    Image.fromarray(np.squeeze(mask))\
                        .save(out_dir / fname.name)
                bar()

        print(f"[Done] 所有mask已保存到: {out_dir}")


# -----------------------------命令行解析--------------------------------------
def parse_args() -> argparse.Namespace:
    TASK_CHOICES = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>",'<OD>','<DENSE_REGION_CAPTION>','<REGION_PROPOSAL>',
                    '<CAPTION_TO_PHRASE_GROUNDING>','<REFERRING_EXPRESSION_SEGMENTATION>','<REGION_TO_SEGMENTATION>','<OPEN_VOCABULARY_DETECTION>','<REGION_TO_CATEGORY>']

    p = argparse.ArgumentParser(
        description="SAM-2 视频人物分割（Florence-2 自动生成人框）")
    p.add_argument("--video", required=True,
                   help="包含多视角视频的根目录")
    p.add_argument("--output", default=None,
                   help="掩膜输出目录 ")
    p.add_argument("--task", default='<CAPTION_TO_PHRASE_GROUNDING>',
                   choices=TASK_CHOICES,help=f"要执行的分割任务，可选值: {', '.join(TASK_CHOICES)}")
    p.add_argument("--text", default=None,
                   help="描述文本，用于 Florence-2 推理")
    p.add_argument("--ann-frame-idx", type=int, default=0,
                   help="与 SAM-2 交互的帧序号")
    p.add_argument("--ann-obj-id", type=int, default=4,
                   help="对象 ID（同一个视频不同物体需不同 ID）")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="Florence-2 推理设备")
    args = p.parse_args()

    return args


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run(parse_args())
