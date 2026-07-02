import os
import zipfile
import pandas as pd
from PIL import Image

def validate_and_pack(submission_dir, test_meta_dir, output_zip):
    """
    Tự động quét qua tất cả các scene đã render, đối chiếu kích thước 
    với file test_poses.csv gốc và đóng gói file zip chuẩn format.
    """
    all_valid = True
    print("[INFO] Dang kiem tra cau truc thu muc va kich thuoc anh...")
    
    # Duyệt qua các thư mục scene tương ứng với metadata kiểm thử
    for scene_id in sorted(os.listdir(test_meta_dir)):
        # Try both direct test_poses.csv or nested under a test/ folder
        csv_path = os.path.join(test_meta_dir, scene_id, "test", "test_poses.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(test_meta_dir, scene_id, "test_poses.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        scene_output_dir = os.path.join(submission_dir, scene_id)
        
        if not os.path.exists(scene_output_dir):
            # Fallback in case the output folder itself is just the scene
            if len(os.listdir(submission_dir)) == 0:
                print(f"[ERROR] Thieu thu muc ket qua cho: {scene_id}")
                all_valid = False
                continue
            scene_output_dir = submission_dir
            
        # Kiểm tra từng file ảnh yêu cầu trong CSV
        for _, row in df.iterrows():
            img_name = row['image_name']
            # Append .png if not present
            if not img_name.endswith('.png'):
                img_name = f"{img_name}.png"
                
            img_path = os.path.join(scene_output_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"[ERROR] Thieu anh: {img_path}")
                all_valid = False
                continue
                
            # Kiểm tra resolution (Rất quan trọng!)
            with Image.open(img_path) as img:
                w, h = img.size
                if w != int(row['width']) or h != int(row['height']):
                    print(f"[ERROR] Sai kich thuoc tai {img_name}: Yeu cau {row['width']}x{row['height']}, Nhan duoc {w}x{h}")
                    all_valid = False

    if not all_valid:
        print("[WARNING] Phat hien loi dinh dang! Vui long sua truoc khi nop bai.")
        return False

    print("[SUCCESS] Moy thu hoan hao! Dang tien hanh nen file submission.zip...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk submission_dir and zip all files
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                if file.endswith('.png'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, submission_dir)
                    zipf.write(full_path, rel_path)
    print(f"[SUCCESS] Da tao thanh cong {output_zip}. San sang nop bai!")
    return True

if __name__ == "__main__":
    # Standard paths matching the competition format
    validate_and_pack("./submission_renders", "./data", "submission.zip")
