import os
import sys
import zipfile
import pandas as pd
from PIL import Image
import numpy as np

def validate_and_zip(output_dir, test_poses_dir, zip_name):
    # Find all test_poses.csv in the test_poses_dir
    # Usually we can find them under test_poses_dir/scene_xxx/test/test_poses.csv
    # Or test_poses_dir is a root directory containing scenes, e.g. test_poses_dir/scene_001/test/test_poses.csv
    # Let's search for test_poses.csv recursively or in child folders.
    
    scenes_checked = 0
    images_checked = 0
    errors = []
    
    # We will accumulate the list of files to zip
    files_to_zip = []
    
    # Let's list all subdirectories in test_poses_dir (each directory is a scene)
    if not os.path.exists(test_poses_dir):
        print(f"Error: test_poses_dir {test_poses_dir} does not exist.")
        sys.exit(1)
        
    scene_folders = [f for f in os.listdir(test_poses_dir) if os.path.isdir(os.path.join(test_poses_dir, f))]
    scene_folders = sorted(scene_folders)
    
    if len(scene_folders) == 0:
        # Maybe test_poses_dir is itself the single scene folder or we have a single test_poses.csv?
        # Let's check if there is a test/test_poses.csv directly
        direct_csv = os.path.join(test_poses_dir, "test", "test_poses.csv")
        if os.path.exists(direct_csv):
            # It's a single scene
            scene_folders = ["."]
            
    print(f"Found {len(scene_folders)} potential scene directories.")
    
    for scene in scene_folders:
        scene_path = os.path.join(test_poses_dir, scene) if scene != "." else test_poses_dir
        csv_path = os.path.join(scene_path, "test", "test_poses.csv")
        
        if not os.path.exists(csv_path):
            # Skip folders that don't contain test_poses.csv
            continue
            
        scene_name = os.path.basename(os.path.abspath(scene_path))
        print(f"\nValidating scene: {scene_name}")
        scenes_checked += 1
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            errors.append(f"Scene {scene_name}: Failed to read CSV {csv_path}: {e}")
            continue
            
        # Target output folder for this scene
        # The structure in output_dir: output_dir/scene_name/image_name
        # Or if scene_folders is "." it's output_dir/image_name
        scene_output_dir = os.path.join(output_dir, scene_name) if scene != "." else output_dir
        
        for row in df.itertuples():
            img_name = row.image_name
            target_w = int(row.width)
            target_h = int(row.height)
            
            img_path = os.path.join(scene_output_dir, img_name)
            
            # Check existence
            if not os.path.exists(img_path):
                errors.append(f"Scene {scene_name}: Missing rendered image: {img_path}")
                continue
                
            # Verify dimensions and integrity
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w != target_w or h != target_h:
                        errors.append(f"Scene {scene_name}: Dimension mismatch for {img_name}. Expected {target_w}x{target_h}, got {w}x{h}")
                    
                    # Check if image is blank (all same color or mean value < 1.0)
                    img_data = np.array(img)
                    if np.all(img_data == img_data[0, 0, 0]) or np.mean(img_data) < 1.0:
                        errors.append(f"Scene {scene_name}: Image {img_name} appears entirely blank or single-colored (mean < 1.0).")
                        
                    # Check for NaNs
                    if np.isnan(img_data).any():
                        errors.append(f"Scene {scene_name}: Image {img_name} contains NaN values.")
            except Exception as e:
                errors.append(f"Scene {scene_name}: Failed to open/verify image {img_name}: {e}")
                continue
                
            images_checked += 1
            # Add to zip list (file path on disk, and relative path inside zip)
            # Relative path should be: scene_name/image_name
            rel_zip_path = os.path.join(scene_name, img_name) if scene != "." else img_name
            files_to_zip.append((img_path, rel_zip_path))
            
    print("\n" + "="*40)
    print("VALIDATION SUMMARY")
    print(f"Scenes checked: {scenes_checked}")
    print(f"Images verified: {images_checked}")
    print(f"Errors found: {len(errors)}")
    print("" + "="*40)
    
    if len(errors) > 0:
        print("\nERRORS DETECTED:")
        for err in errors[:15]:
            print(f" - {err}")
        if len(errors) > 15:
            print(f" ... and {len(errors) - 15} more errors.")
        print("\nSubmission ZIP will NOT be created due to validation errors.")
        sys.exit(1)
    else:
        print("\nAll checks passed! Packaging submission.zip...")
        zip_out_path = os.path.join(output_dir, zip_name)
        
        with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath, relpath in files_to_zip:
                zipf.write(filepath, relpath)
                
        print(f"Successfully packaged {len(files_to_zip)} images into {zip_out_path}")
        print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate output folder structure and pack submission")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the rendered outputs")
    parser.add_argument("--test_poses_dir", type=str, required=True, help="Root directory containing input scenes/test_poses.csv")
    parser.add_argument("--zip_name", type=str, default="submission.zip", help="Name of the output zip file")
    args = parser.parse_args()
    
    validate_and_zip(args.output_dir, args.test_poses_dir, args.zip_name)
