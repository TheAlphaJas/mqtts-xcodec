import gdown
import os
import shutil
import subprocess

# === Define Google Drive file IDs and target dirs ===
files = {
    "config.json": ("1RkzZdSuXzWtSeAKccH45hGPUIusJLsCX", "ckpt/"),
    "g_00600000.ckpt": ("1XOtGWUlem8cG6PCyTyjHHtC5VvUogvlE", "quantizer/checkpoints/"),
    "last.ckpt": ("1JtzeY3kGVks1O1NgwwrYPyPxfDli9kc8", "ckpt/")
}

# === Create directories if missing ===
for _, (_, target_dir) in files.items():
    os.makedirs(target_dir, exist_ok=True)

# === Download and move files ===
for file_name, (file_id, target_dir) in files.items():
    dest_path = os.path.join(target_dir, file_name)
    if os.path.exists(dest_path):
        print(f"✅ {file_name} already exists, skipping download.")
        continue

    print(f"⬇️  Downloading {file_name}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    temp_path = file_name
    gdown.download(url, temp_path, quiet=False)
    shutil.move(temp_path, dest_path)
    print(f"📦 Moved {file_name} → {dest_path}")

print("\n✅ All files downloaded and organized successfully!")
