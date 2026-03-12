import cv2
import gradio as gr
import numpy as np
import os

points_src = []
points_dst = []
image = None


def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = np.array(img) if img is not None else None
    return image


def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    if image is None:
        return None

    x, y = int(evt.index[0]), int(evt.index[1])

    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 2, (255, 0, 0), -1)
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 2, (0, 0, 255), -1)

    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image


def _compute_rbf_weights(source_pts: np.ndarray, displacement: np.ndarray, eps: float) -> np.ndarray:
    n = source_pts.shape[0]
    diff = source_pts[:, None, :] - source_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    k = np.sqrt(dist2 + eps)
    k += np.eye(n, dtype=np.float32) * (1e-6 + eps)
    return np.linalg.solve(k, displacement.astype(np.float32))


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    if image is None:
        return None

    img = np.array(image)
    h, w = img.shape[:2]

    source_pts = np.asarray(source_pts, dtype=np.float32)
    target_pts = np.asarray(target_pts, dtype=np.float32)

    if source_pts.ndim != 2 or target_pts.ndim != 2:
        return img
    if source_pts.shape[0] == 0 or source_pts.shape[0] != target_pts.shape[0]:
        return img

    displacement_inv = source_pts - target_pts
    weights = _compute_rbf_weights(target_pts, displacement_inv, eps)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    diff = grid[:, None, :] - target_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)

    beta = float(alpha)
    phi = (dist2 + eps) ** (0.5 * beta)
    disp = phi @ weights

    src_pos = grid + disp
    map_x = np.clip(src_pos[:, 0], 0, w - 1).reshape(h, w).astype(np.float32)
    map_y = np.clip(src_pos[:, 1], 0, h - 1).reshape(h, w).astype(np.float32)

    warped_image = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped_image


def run_warping():
    global points_src, points_dst, image
    if image is None:
        return None
    return point_guided_deformation(image, np.array(points_src), np.array(points_dst))


def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)


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
    demo.launch(server_name="127.0.0.1", share=False)
