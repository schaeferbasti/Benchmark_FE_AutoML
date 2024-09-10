import os
import shutil

for filename in os.listdir():
    if filename.endswith(".parquet"):
        # Split the filename to extract type, dataset, method, and fold
        parts = filename.split('_')
        method = parts[-2]
        parts.remove(method)
        parts.remove(parts[-1])
        dataset = "_".join(parts)

        # Create a new folder path based on dataset and method
        new_folder = f"{dataset}_{method}"

        # Ensure the folder exists
        os.makedirs(new_folder, exist_ok=True)

        # Define the source and destination paths
        source_path = os.path.join(filename)
        destination_path = os.path.join(new_folder, filename)

        # Move the file to the new folder
        shutil.move(source_path, destination_path)
        print(f"Moved {filename} to {new_folder}")
