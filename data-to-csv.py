import os
import csv

bucket_path = "gs://mlops_99/destination"
local_folder = "data/raw"  # If you still have local files
csv_file = "labels.csv"

with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    for root, dirs, files in os.walk(local_folder):
        label = os.path.basename(root)
        for file in files:
            gcs_path = f"{bucket_path}/{label}/{file}"
            writer.writerow([gcs_path, label])
