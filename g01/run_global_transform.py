import cv2
import gradio as gr
import numpy as np
import os


def to_3x3(affine_matrix: np.ndarray) -> np.ndarray:
    return np.vstack([affine_matrix, [0.0, 0.0, 1.0]]).astype(np.float32)


def apply_transform(
    image,
    scale,
    rotation,
    translation_x,
    translation_y,
    flip_horizontal,
):
    if image is None:
        return None

    image = np.array(image)

    pad_size = min(image.shape[0], image.shape[1]) // 2
    if image.ndim == 2:
        image_new = np.full(
            (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1]),
            255,
            dtype=np.uint8,
        )
    else:
        image_new = np.full(
            (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], image.shape[2]),
            255,
            dtype=np.uint8,
        )

    image_new[pad_size : pad_size + image.shape[0], pad_size : pad_size + image.shape[1]] = image
    image = image_new

    h, w = image.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    t_neg_center = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    t_pos_center = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    s = np.array([[float(scale), 0.0, 0.0], [0.0, float(scale), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    theta = np.deg2rad(float(rotation))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    r = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    if flip_horizontal:
        f = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        f = np.eye(3, dtype=np.float32)

    t_translation = np.array(
        [[1.0, 0.0, float(translation_x)], [0.0, 1.0, float(translation_y)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    m = t_translation @ (t_pos_center @ r @ s @ f @ t_neg_center)

    transformed_image = cv2.warpAffine(
        image,
        m[:2, :],
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if image.ndim == 3 else 255,
    )

    return transformed_image


def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            image_output = gr.Image(label="Transformed Image")

        inputs = [
            image_input,
            scale,
            rotation,
            translation_x,
            translation_y,
            flip_horizontal,
        ]

        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


if __name__ == "__main__":
    no_proxy_items = {"127.0.0.1", "localhost"}
    existing = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    if existing:
        for item in existing.split(","):
            item = item.strip()
            if item:
                no_proxy_items.add(item)
    no_proxy_value = ",".join(sorted(no_proxy_items))
    os.environ["NO_PROXY"] = no_proxy_value
    os.environ["no_proxy"] = no_proxy_value
    interactive_transform().launch(server_name="127.0.0.1", share=False)
