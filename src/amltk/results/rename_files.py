import os

files = os.listdir(os.curdir)

# Filter out the parquet files that do not contain "tabular_data"
parquet_files = [file for file in files if file.endswith(".parquet") and "tabular_data" not in file]

for file in parquet_files:
    # Split the filename and add 'lgbm' before the fold number
    parts = file.split("_")
    parts[4] = ''.join([i for i in parts[4] if not i.isdigit()])

    new_file_name = "_".join(parts[:-1]) + "_lgbm_" + parts[-1]
    print(new_file_name)
    # Rename the file
    os.rename(file, new_file_name)
