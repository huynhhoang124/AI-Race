import os
from PIL import Image

def generate_mock_data():
    base_dir = "mock_data/scene_test"
    train_dir = os.path.join(base_dir, "train")
    images_dir = os.path.join(train_dir, "images")
    sparse_dir = os.path.join(train_dir, "sparse/0")
    test_dir = os.path.join(base_dir, "test")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Create mock images
    Image.new("RGB", (64, 64), (255, 0, 0)).save(os.path.join(images_dir, "0001.png"))
    Image.new("RGB", (64, 64), (0, 255, 0)).save(os.path.join(images_dir, "0002.png"))
    
    # 2. Create cameras.txt
    # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    # PINHOLE parameters: fx, fy, cx, cy
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE 64 64 50.0 50.0 32.0 32.0\n")
        
    # 3. Create images.txt
    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # POINTS2D[] as (X, Y, POINT3D_ID)
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 0001.png\n")
        f.write("10.0 10.0 1\n")
        f.write("2 1.0 0.0 0.0 0.0 0.0 0.0 1.0 1 0002.png\n")
        f.write("10.0 10.0 1\n")
        
    # 4. Create points3D.txt
    # POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("1 0.0 0.0 1.0 255 0 0 0.1 1 0 2 0\n")
        
    # 5. Create test_poses.csv
    # image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
    with open(os.path.join(test_dir, "test_poses.csv"), "w") as f:
        f.write("image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n")
        f.write("0003.png,1.0,0.0,0.0,0.0,0.0,0.0,2.0,50.0,50.0,32.0,32.0,64,64\n")
        
    print("Mock dataset created successfully at ./mock_data")

if __name__ == "__main__":
    generate_mock_data()
