import cv2
import numpy as np
import os
import glob
import subprocess # 用于调用 ffmpeg
import shutil     # 用于检查 ffmpeg 是否存在

"""
Flash‑frame detector (revised)
=============================
核心逻辑
---------
* **判定闪光帧**：
  1. 当帧的最大白色连通域面积 ≥ `min_white_cluster_area`。
  2. 当帧的白色像素数比 *前*、*后* 两帧均高 `white_gain`（默认 30 %）。
* **重试策略**（若首次未命中）：
  1. `min_white_cluster_area` 每次固定减 **10 000** 像素。
  2. `white_gain` 每次乘 **0.95**（降低 5 %）。 # 注意：下方 _retry_params 的实现与此描述不同
  3. 任何阈值降至 ≤ 0 时终止重试。
"""


def _frame_stats(frame: np.ndarray, brightness_threshold: int):
    """Return white‑pixel count and largest white‑cluster area for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    white_pixels = int(np.sum(thresh == 255))
    largest_cluster_area = int(max(stats[1:, cv2.CC_STAT_AREA], default=0))
    return white_pixels, largest_cluster_area


def find_flash_frame(
    cap: cv2.VideoCapture,
    brightness_threshold: int = 240,
    min_white_cluster_area: int = 500,
    white_gain: float = 0.30,
    debug: bool = False,
):
    """遍历视频帧，返回符合闪光条件的帧索引。

    判定：
      • 最大白色连通域面积 ≥ `min_white_cluster_area`;
      • 白色像素数 > `white_gain` × 前帧 & 后帧的白像素数。
    """
    # 初始化三帧滑动窗口 prev → curr → next
    ret, prev_frame = cap.read()
    if not ret:
        return -1
    prev_white_px, prev_cluster = _frame_stats(prev_frame, brightness_threshold)

    ret, curr_frame = cap.read()
    if not ret:
        return -1
    curr_white_px, curr_cluster = _frame_stats(curr_frame, brightness_threshold)

    frame_index = 1  # 当前帧索引 (curr_frame 对应的 OpenCV 帧索引从0开始，这里是第二帧，所以是索引1)
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        # next_cluster 在原始代码中未被使用，所以这里也只获取 next_white_px
        next_white_px, _ = _frame_stats(next_frame, brightness_threshold)
        if debug:
            print(
                f"Frame {frame_index}: cluster={curr_cluster}, white_px={curr_white_px}"
            )

        if (
            (curr_cluster >= min_white_cluster_area and curr_white_px > (1+white_gain) * prev_white_px)
            or
            (curr_cluster >= min_white_cluster_area and curr_white_px > (1+white_gain) * next_white_px)
        ):
            if debug:
                gain_prev_str = f"{curr_white_px/prev_white_px:.2f}" if prev_white_px > 0 else ("inf" if curr_white_px > 0 else "N/A")
                gain_next_str = f"{curr_white_px/next_white_px:.2f}" if next_white_px > 0 else ("inf" if curr_white_px > 0 else "N/A")
                print(
                    f"Flash detected at frame {frame_index} | cluster={curr_cluster} | "
                    f"gain_prev={gain_prev_str} | "
                    f"gain_next={gain_next_str}"
                )
            return frame_index

        # 滑窗右移
        prev_white_px, prev_cluster = curr_white_px, curr_cluster
        # curr_frame = next_frame # OpenCV的cap.read()会移动指针，所以curr_frame的更新在下一次循环的cap.read()
        curr_white_px, curr_cluster = _frame_stats(next_frame, brightness_threshold) # 更新curr为next的统计数据
        frame_index += 1

    return -1


def _retry_params(min_white_cluster_area: int, white_gain: float):
    """按用户要求衰减阈值：面积 - 10 000，增益 × 0.95""" 
    return max(0, min_white_cluster_area - 10000), max(0.1, white_gain - 0.05)


def trim_video( # 函数名保持不变，但内部实现将使用ffmpeg
    input_path: str,
    output_path: str,
    brightness_threshold: int = 240,
    min_white_cluster_area: int = 500,
    white_gain: float = 0.30,
    debug: bool = False,
):
    """
    检测视频中的闪光帧，并使用 ffmpeg 从闪光帧的下一帧开始无损裁剪视频并保存。
    如果未检测到闪光帧，则不创建输出文件并返回 -1。
    """
    if shutil.which("ffmpeg") is None:
        print("错误：未找到 ffmpeg。请确保已安装 ffmpeg 并将其添加至系统 PATH。")
        return -1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件 {input_path}")
        return -1

    # find_flash_frame 会从 cap 的当前位置开始读取。
    # 对于首次调用或每次重试（如果 cap 被重新打开或重置），它会从头开始。
    # 注意：原始的 find_flash_frame 不会重置 cap 的读取位置，
    # 但由于 trim_video 每次都打开新的 cap，所以 find_flash_frame 总是从头开始。
    flash_idx = find_flash_frame(
        cap,
        brightness_threshold,
        min_white_cluster_area,
        white_gain,
        debug,
    )

    if flash_idx == -1:
        if debug:
            print(f"在 {input_path} 中未检测到闪光帧 (参数: area={min_white_cluster_area}, gain={white_gain:.2f})。")
        cap.release()
        return -1

    # 获取视频属性用于计算时间戳
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() # OpenCV 处理完毕，释放 cap

    if fps == 0: # total_frames 可能为0或不准确，但fps为0是硬伤
        print(f"错误: 视频 {input_path} 的 FPS 为 0。无法计算时间戳进行裁剪。")
        return -1

    # 我们要从闪光帧的 *下一帧* 开始保留
    start_frame_to_keep = flash_idx + 1

    if start_frame_to_keep >= total_frames and total_frames > 0: # total_frames > 0 以免fps有效但total_frames未知的情况
        if debug:
            print(f"闪光帧 ({flash_idx}) 是最后一帧或已超出视频总帧数 ({total_frames})。"
                  f"裁剪后的视频 {output_path} 将为空或不创建。")
        # 确保如果输出文件已存在（可能来自上次失败的尝试），则将其删除
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError as e: print(f"警告：无法删除旧的空输出文件 {output_path}: {e}")
        return flash_idx # 返回检测到的 flash_idx，即使没有内容可写

    start_time = start_frame_to_keep / fps

    # 构建 ffmpeg 命令
    command = [
        "ffmpeg",
        "-y",  # 无需确认即覆盖输出文件
        "-i", input_path,
        "-ss", str(start_time), # 从计算出的开始时间
        "-c", "copy",          # 直接复制流，不重新编码 (无损)
        "-avoid_negative_ts", "make_zero", # 处理可能的起始时间戳问题
        output_path,
    ]

    if debug:
        print(f"执行 ffmpeg 命令: {' '.join(command)}")

    try:
        # 使用 text=True (或 universal_newlines=True) 使 stdout/stderr 为字符串
        # 指定编码和错误处理方式
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if debug:
            print(f"ffmpeg STDOUT for {output_path}:\n{process.stdout}")
            if process.stderr:
                 print(f"ffmpeg STDERR for {output_path}:\n{process.stderr}")
            print(f"无损剪切后视频保存至 {output_path}")
        return flash_idx # 成功，返回检测到的闪光帧索引
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 执行失败 for {input_path}, 返回码: {e.returncode}")
        print(f"ffmpeg STDOUT:\n{e.stdout}")
        print(f"ffmpeg STDERR:\n{e.stderr}")
        # 如果 ffmpeg 失败，删除可能已创建的损坏的输出文件
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass
        return -1 # 返回 -1 表示裁剪失败（即使闪光帧已找到）
    except FileNotFoundError: # shutil.which 应该能更早捕获，但作为保险
        print("错误：ffmpeg 命令未找到。请确保已安装并配置在系统路径中。")
        return -1


def process_videos_in_folder(
    folder_path: str,
    brightness_threshold: int = 228,
    min_white_cluster_area: int = 150000,
    white_gain: float = 0.30,
    debug: bool = False,
):
    video_files = glob.glob(os.path.join(folder_path, "*.MP4"))
    save_dir = os.path.join(folder_path, "videos") # 修改输出文件夹名
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "flash_frame_index.txt")
    print(f"记录写入 {log_path}")

    failed = [] # 用于存储第一次处理失败的视频信息 (vp, outp, base)
    # 初次处理所有视频
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"[Initial Pass Parameters: area={min_white_cluster_area}, gain={white_gain:.4f}]\n")
        for vp_idx, vp in enumerate(video_files):
            base = os.path.basename(vp)
            # 尝试从文件名提取数字前缀，否则使用不带扩展名的基本名
            try:
                idx_prefix = f"{int(base.split('_')[0]):02d}"
            except (ValueError, IndexError):
                idx_prefix = os.path.splitext(base)[0]
            
            outp = os.path.join(save_dir, f"{idx_prefix}.mp4")
            print(f"\n--- 处理 ({vp_idx+1}/{len(video_files)}): {vp} ---")
            flash_idx = trim_video(
                vp,
                outp,
                brightness_threshold,
                min_white_cluster_area, # 使用初始参数
                white_gain,             # 使用初始参数
                debug,
            )
            log.write(f"{base}: {flash_idx} (Initial)\n")
            log.flush()
            if flash_idx == -1: # 包括 ffmpeg 执行失败的情况
                failed.append((vp, outp, base))

    # 重试逻辑
    current_min_area = min_white_cluster_area
    current_gain = white_gain
    retry_attempt = 0

    while failed:
        retry_attempt += 1
        print(f"\n--- 重试 第 {retry_attempt} 轮未命中视频 ({len(failed)} 个) ---")
        
        # 更新参数前，先获取用于本次重试的参数
        retry_min_area, retry_gain = _retry_params(current_min_area, current_gain)
        if retry_min_area <= 0 or retry_gain <= 0.001: # 比较浮点数用一个小的epsilon
            print(f"阈值已降至零或以下 (area={retry_min_area}, gain={retry_gain:.4f})，终止重试。")
            break
        
        current_min_area, current_gain = retry_min_area, retry_gain # 应用衰减后的参数

        next_failed = []
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(
                f"\n[Retry Attempt {retry_attempt} with area={current_min_area}, gain={current_gain:.4f}]\n"
            )
            for vp, outp, base in failed:
                print(f"[Retry {retry_attempt}] {vp}")
                flash_idx = trim_video(
                    vp,
                    outp,
                    brightness_threshold,
                    current_min_area, # 使用衰减后的参数
                    current_gain,     # 使用衰减后的参数
                    debug,
                )
                log.write(f"{base}: {flash_idx} (Retry {retry_attempt})\n")
                log.flush()
                if flash_idx == -1:
                    next_failed.append((vp, outp, base))
        failed = next_failed
        if not failed:
            print("所有先前失败的视频已在重试中处理完毕或找到闪光帧。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="裁剪视频，去除闪光帧前内容（使用ffmpeg无损裁剪）")
    parser.add_argument("--video", type=str, help="单个视频路径")
    parser.add_argument("--folder", type=str, help="批处理文件夹路径")
    parser.add_argument("--brightness_threshold", type=int, default=228)
    parser.add_argument("--min_white_cluster_area", type=int, default=150000)
    parser.add_argument("--white_gain", type=float, default=0.30, help="初始相邻帧增益阈值 (30%% = 0.30)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("关键错误：未在系统中找到 ffmpeg。此脚本需要 ffmpeg 进行无损视频裁剪。")
        print("请安装 ffmpeg 并确保它在系统的 PATH 环境变量中。")
        sys.exit(1) # 使用 sys.exit

    if args.video:
        video_dir = os.path.dirname(args.video) if os.path.dirname(args.video) else "."
        save_dir = os.path.join(video_dir, "videos") # 和文件夹处理用一样的输出目录名
        os.makedirs(save_dir, exist_ok=True)
        
        base = os.path.basename(args.video)
        try:
            idx_prefix = f"{int(base.split('_')[0]):02d}"
        except (ValueError, IndexError):
            idx_prefix = os.path.splitext(base)[0]
        outp = os.path.join(save_dir, f"{idx_prefix}.mp4")
        
        print(f"--- 处理单个视频: {args.video} ---")
        current_area, current_gain = args.min_white_cluster_area, args.white_gain
        flash = -1
        retry_count = 0
        
        # 对单个视频也应用重试逻辑
        while True:
            print(f"尝试参数: area={current_area}, gain={current_gain:.4f}")
            flash = trim_video(
                args.video,
                outp,
                args.brightness_threshold,
                current_area,
                current_gain,
                args.debug,
            )
            if flash != -1:
                print(f"处理完成。闪光帧位于索引 {flash}。裁剪后的视频 (如果不为空) 已保存至 {outp}")
                break
            
            retry_count += 1
            print(f"未命中，准备重试 {retry_count}...")
            prev_area, prev_gain = current_area, current_gain
            current_area, current_gain = _retry_params(current_area, current_gain)
            if current_area <= 0 or current_gain <= 0.001:
                print(f"参数已降至零或以下 (area={current_area}, gain={current_gain:.4f})，终止对 {args.video} 的重试。")
                print(f"最终未能检测到 {args.video} 中的闪光帧。")
                break
            if not args.debug: # 如果不是debug模式，只在第一次重试前打印，之后不再打印
                if retry_count > 1: continue
            
    elif args.folder:
        process_videos_in_folder(
            args.folder,
            args.brightness_threshold,
            args.min_white_cluster_area,
            args.white_gain,
            args.debug,
        )
    else:
        parser.print_help() # 使用 parser.print_help()
        print("\n请使用 --video 或 --folder 指定输入。")