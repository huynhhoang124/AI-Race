import os
import sys
import math
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat
import cv2

def match_histograms_1d(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)

def match_histograms_rgb(source, template):
    # source and template: float32 arrays in range [0, 1]
    matched = np.zeros_like(source)
    for channel in range(3):
        matched[..., channel] = match_histograms_1d(source[..., channel], template[..., channel])
    return matched

def apply_unsharp_mask(image, sigma=1.0, strength=0.8):
    # image is a numpy array of shape (H, W, 3) in range [0, 1]
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = float(1.0 + strength) * image - float(strength) * blurred
    return np.clip(sharpened, 0.0, 1.0)

def load_train_cameras_positions(model_path):
    import json
    cameras_json_path = os.path.join(model_path, "cameras.json")
    if not os.path.exists(cameras_json_path):
        print(f"[Warning] cameras.json not found at {cameras_json_path}. Skipping color matching.")
        return None, None
        
    with open(cameras_json_path, "r") as f:
        cams_info = json.load(f)
        
    positions = []
    img_names = []
    for cam in cams_info:
        positions.append(cam["position"])
        img_names.append(cam["img_name"])
    return np.array(positions), img_names

def find_nearest_image_path(camera_pos, train_positions, train_img_names, source_path):
    if train_positions is None or len(train_positions) == 0:
        return None
    dists = np.linalg.norm(train_positions - camera_pos, axis=1)
    nearest_idx = np.argmin(dists)
    nearest_img_name = train_img_names[nearest_idx]
    
    # Try multiple subdirectories for training images
    possible_paths = [
        os.path.join(source_path, "images", nearest_img_name),
        os.path.join(source_path, "images_2", nearest_img_name),
        os.path.join(source_path, "images_4", nearest_img_name),
        os.path.join(source_path, "images_8", nearest_img_name)
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def render_test_poses(model_path, test_poses_path, output_dir, pipeline, background, source_path, do_postprocess):
    # Load test poses
    df = pd.read_csv(test_poses_path)
    
    # Initialize Gaussian Model
    gaussians = GaussianModel(args.sh_degree)
    
    # Find latest iteration
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(point_cloud_dir):
        sys.exit(f"Error: Point cloud directory {point_cloud_dir} does not exist. Ensure training completed successfully.")
        
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(point_cloud_dir) if fname.startswith("iteration_")]
    if len(saved_iters) == 0:
        sys.exit("Error: No saved iterations found in point cloud folder.")
        
    latest_iter = max(saved_iters)
    ply_path = os.path.join(point_cloud_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    print(f"Loading point cloud at iteration {latest_iter} from {ply_path}")
    gaussians.load_ply(ply_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training camera positions for color matching
    train_positions, train_img_names = None, None
    if do_postprocess and source_path:
        train_positions, train_img_names = load_train_cameras_positions(model_path)
        
    # Check separate SH availability
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        separate_sh = True
    except:
        separate_sh = False
        
    for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Rendering test poses")):
        image_name = row.image_name
        qw, qx, qy, qz = row.qw, row.qx, row.qy, row.qz
        tx, ty, tz = row.tx, row.ty, row.tz
        fx, fy = row.fx, row.fy
        cx, cy = row.cx, row.cy
        width, height = int(row.width), int(row.height)
        
        # Calculate R and T
        qvec = np.array([qw, qx, qy, qz])
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array([tx, ty, tz])
        
        # Calculate FoV
        FoVx = 2.0 * math.atan(width / (2.0 * fx))
        FoVy = 2.0 * math.atan(height / (2.0 * fy))
        
        # Create dummy image to satisfy Camera constructor
        dummy_image = Image.new("RGB", (width, height), (0, 0, 0))
        
        cam = Camera(
            resolution=(width, height),
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params=None,
            image=dummy_image,
            invdepthmap=None,
            image_name=image_name,
            uid=idx,
            data_device=args.data_device,
            train_test_exp=False,
            is_test_dataset=False,
            is_test_view=True,
            cx=cx,
            cy=cy,
            fx=fx,
            fy=fy
        )
        
        # Render image
        rendering_pkg = render(
            cam,
            gaussians,
            pipeline,
            background,
            use_trained_exp=False,
            separate_sh=separate_sh
        )
        rendered_tensor = rendering_pkg["render"]  # (3, H, W) on GPU
        
        # Convert to numpy array (H, W, 3) in float32
        rendered_np = rendered_tensor.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        
        # Apply Post-processing
        if do_postprocess:
            # Color/Histogram Matching
            if train_positions is not None and source_path:
                # Find target camera center location
                camera_center = cam.camera_center.cpu().numpy()
                nearest_img_path = find_nearest_image_path(camera_center, train_positions, train_img_names, source_path)
                if nearest_img_path:
                    try:
                        template_img = Image.open(nearest_img_path).convert("RGB").resize((width, height))
                        template_np = np.array(template_img).astype(np.float32) / 255.0
                        rendered_np = match_histograms_rgb(rendered_np, template_np)
                    except Exception as e:
                        print(f"Error during histogram matching for {image_name}: {e}")
            
            # Sharpening
            rendered_np = apply_unsharp_mask(rendered_np, sigma=1.0, strength=0.08)
            
        # Convert back to PIL Image and save
        rendered_img_uint8 = (rendered_np * 255.0).astype(np.uint8)
        pil_out = Image.fromarray(rendered_img_uint8)
        
        output_image_path = os.path.join(output_dir, image_name)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        pil_out.save(output_image_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Inference renderer for test camera poses")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--test_poses_path", type=str, required=True, help="Path to test_poses.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save images")
    parser.add_argument("--skip_postprocess", action="store_true", help="Skip post-processing step (histogram matching & sharpening)")
    
    args = get_combined_args(parser)
    
    # Initialize system seed & state
    from utils.general_utils import safe_state
    safe_state(args.quiet)
    
    # Set background color
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_test_poses(
        model_path=args.model_path,
        test_poses_path=args.test_poses_path,
        output_dir=args.output_dir,
        pipeline=pipeline.extract(args),
        background=background,
        source_path=args.source_path,
        do_postprocess=not args.skip_postprocess
    )
