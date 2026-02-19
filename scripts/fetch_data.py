import kagglehub
import os
import shutil

# Download dataset via kagglehub
dataset_path = kagglehub.dataset_download(
    "yasserh/wine-quality-dataset"
)

# Identify the CSV file
src_file = os.path.join(dataset_path, "WineQT.csv")

# Destination inside your project
dest_dir = os.path.join("data", "raw")
os.makedirs(dest_dir, exist_ok=True)

dest_file = os.path.join(dest_dir, "WineQT.csv")

# Copy file into project
shutil.copy(src_file, dest_file)

print("Dataset copied to:", dest_file)
