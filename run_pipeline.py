import os
import sys
import subprocess
import argparse

def check_command_exists(cmd):
    import shutil
    return shutil.which(cmd) is not None

def run_command(command, cwd=None):
    print(f"\nRunning command: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def setup_depth_anything_v2():
    if not os.path.exists("Depth-Anything-V2"):
        print("\nDepth-Anything-V2 directory not found. Cloning repository...")
        clone_cmd = ["git", "clone", "https://github.com/DepthAnything/Depth-Anything-V2.git"]
        run_command(clone_cmd)
        
    checkpoint_dir = os.path.join("Depth-Anything-V2", "checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, "depth_anything_v2_vitl.pth")
    if not os.path.exists(checkpoint_file):
        print(f"\n[Warning] Depth Anything v2 weights not found at {checkpoint_file}.")
        print("Please download the vitl checkpoint from Hugging Face and place it in Depth-Anything-V2/checkpoints/")
        print("Continuing without depth maps...")
        return False
    return True

def process_depth_maps(scene_train_path, depth_out_dir):
    os.makedirs(depth_out_dir, exist_ok=True)
    images_dir = os.path.join(scene_train_path, "images")
    
    # Run Depth Anything v2 prediction
    # Path: Depth-Anything-V2/run.py
    depth_run_cmd = [
        sys.executable,
        "Depth-Anything-V2/run.py",
        "--encoder", "vitl",
        "--pred-only",
        "--grayscale",
        "--img-path", images_dir,
        "--outdir", depth_out_dir
    ]
    print(f"\nGenerating depth maps for {images_dir}...")
    run_command(depth_run_cmd)
    
    # Run scale matching script
    # Path: utils/make_depth_scale.py
    scale_cmd = [
        sys.executable,
        "utils/make_depth_scale.py",
        "--base_dir", scene_train_path,
        "--depths_dir", depth_out_dir
    ]
    print(f"\nMatching depth map scale to COLMAP for {scene_train_path}...")
    run_command(scale_cmd)
    return True

def run_pipeline(data_root, output_dir, model_dir, mode, skip_depth, iterations_override):
    # Locate all scenes
    if not os.path.exists(data_root):
        sys.exit(f"Error: Data root directory {data_root} does not exist.")
        
    scene_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    scene_folders = sorted(scene_folders)
    
    # If the root itself has train and test directories, treat it as a single scene
    is_single_scene = False
    if os.path.exists(os.path.join(data_root, "train")) and os.path.exists(os.path.join(data_root, "test")):
        scene_folders = ["."]
        is_single_scene = True
        
    if len(scene_folders) == 0:
        sys.exit("Error: No scenes found in data root directory.")
        
    print(f"Starting pipeline in {mode.upper()} mode for {len(scene_folders)} scenes...")
    
    # Set default iterations based on mode
    if iterations_override is not None:
        train_iterations = iterations_override
    else:
        train_iterations = 500 if mode == "test" else 30000
        
    # Prepare depth setup if running prod mode and depth is not skipped
    use_depth = False
    if mode == "prod" and not skip_depth:
        if setup_depth_anything_v2():
            use_depth = True
            
    # Iterate through scenes
    for scene in scene_folders:
        scene_path = os.path.join(data_root, scene) if not is_single_scene else data_root
        scene_name = os.path.basename(os.path.abspath(scene_path))
        print(f"\n==========================================")
        print(f"PROCESSING SCENE: {scene_name}")
        print(f"==========================================")
        
        scene_train_path = os.path.join(scene_path, "train")
        scene_test_csv = os.path.join(scene_path, "test", "test_poses.csv")
        
        if not os.path.exists(scene_train_path) or not os.path.exists(scene_test_csv):
            print(f"Warning: Skipping scene {scene_name} due to missing train/ or test/test_poses.csv.")
            continue
            
        scene_model_dir = os.path.join(model_dir, scene_name)
        scene_output_dir = os.path.join(output_dir, scene_name) if not is_single_scene else output_dir
        
        # 1. Depth map estimation
        depth_dir_name = "depths"
        depth_dir_path = os.path.join(scene_train_path, depth_dir_name)
        has_depth_params = False
        
        if use_depth:
            try:
                has_depth_params = process_depth_maps(scene_train_path, depth_dir_path)
            except Exception as e:
                print(f"Failed to generate depth maps: {e}. Training will continue without depth regularization.")
                has_depth_params = False
                
        # 2. Model Training
        # Build training command
        train_cmd = [
            sys.executable,
            "train.py",
            "-s", scene_train_path,
            "-m", scene_model_dir,
            "--iterations", str(train_iterations),
            "--test_iterations", "-1",
            "--save_iterations", str(train_iterations),
            "--quiet"
        ]
        
        if mode == "test":
            train_cmd.extend([
                "-r", "4",
                "--data_device", "cpu"
            ])
        else: # prod mode
            train_cmd.extend([
                "-r", "1",
                "--antialiasing",
                "--densify_grad_threshold", "0.00015"
            ])
            # Check sparse adam optimizer availability
            try:
                from diff_gaussian_rasterization import SparseGaussianAdam
                train_cmd.extend(["--optimizer_type", "sparse_adam"])
            except:
                train_cmd.extend(["--optimizer_type", "default"])
                
            if has_depth_params:
                train_cmd.extend(["-d", depth_dir_name])
                
        print(f"\nTraining model for {scene_name}...")
        run_command(train_cmd)
        
        # 3. Model Rendering
        render_cmd = [
            sys.executable,
            "render_test.py",
            "-s", scene_train_path,
            "-m", scene_model_dir,
            "--test_poses_path", scene_test_csv,
            "--output_dir", scene_output_dir
        ]
        
        if mode == "test":
            render_cmd.append("--skip_postprocess")
            
        print(f"\nRendering test poses for {scene_name}...")
        run_command(render_cmd)
        
    # 4. Packaging and Validation
    validate_cmd = [
        sys.executable,
        "validate_submission.py",
        "--output_dir", output_dir,
        "--test_poses_dir", data_root,
        "--zip_name", "submission.zip"
    ]
    print("\nValidating rendered outputs and zipping submission...")
    run_command(validate_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end training and inference pipeline")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the directory containing input scenes")
    parser.add_argument("--output_dir", type=str, default="./submission_renders", help="Path to render output directory")
    parser.add_argument("--model_dir", type=str, default="./output_models", help="Path to save trained checkpoints")
    parser.add_argument("--mode", type=str, choices=["test", "prod"], default="test", help="test: low-res fast run; prod: SOTA high-res run")
    parser.add_argument("--skip_depth", action="store_true", help="Skip Depth Anything v2 estimation in prod mode")
    parser.add_argument("--iterations", type=int, default=None, help="Override training iterations count")
    args = parser.parse_args()
    
    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        mode=args.mode,
        skip_depth=args.skip_depth,
        iterations_override=args.iterations
    )
