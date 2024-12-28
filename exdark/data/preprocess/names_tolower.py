import os
import sys

dir_path = sys.argv[1]

for root, dirs, files in os.walk(dir_path):
    for file in files:
        file_lower = file.lower()
        file_path = os.path.join(root, file)
        file_path_lower = os.path.join(root, file_lower)
        os.rename(file_path, file_path_lower)