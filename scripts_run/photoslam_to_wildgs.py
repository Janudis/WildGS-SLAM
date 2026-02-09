from pathlib import Path
import shutil

src = Path("/home/dimitris/WildGS-SLAM/datasets/KITTI_seq00_mono_0_1000/rgbs")
dst = Path("/home/dimitris/WildGS-SLAM/datasets/KITTI_seq00_0_1000/rgb")
dst.mkdir(parents=True, exist_ok=True)

files = sorted(src.glob("*.png"))
print("Found", len(files), "images")

for i, f in enumerate(files):
    new_name = dst / f"frame_{i:05d}.png"
    # Just copy with new name; OpenCV will still read it even if it's really JPG
    shutil.copy2(f, new_name)
    # if you prefer to move instead of copy, use: f.rename(new_name)

print("Done.")