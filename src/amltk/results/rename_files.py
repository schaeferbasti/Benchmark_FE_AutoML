import os

files = os.listdir(os.curdir)

# Filter out the parquet files that do not contain "tabular_data"
parquet_files = [file for file in files if file.endswith(".parquet") and "tabular_data" not in file]

for file in parquet_files:
    # Split the filename and add 'lgbm' before the fold number
    parts = file.split("_")
    if parts[-5][-1].isdigit():
        print(parts[-5][-1])
        parts[-5] = parts[-5].replace(parts[-5][-1], "")
        if parts[-5][-1].isdigit():
            print(parts[-5][-1])
            parts[-5] = parts[-5].replace(parts[-5][-1], "")
            print(parts[-5])
        else:
            print(parts[-5])
        parts = "_".join(parts)
        new_file_name = parts
        print(new_file_name)
        # Rename the file
        os.rename(file, new_file_name)
