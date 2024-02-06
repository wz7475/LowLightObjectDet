import os
import shutil
import sys

src_dir = sys.argv[1]
dst_dir = sys.argv[2]

for root, dirs, files in os.walk(src_dir):
    for file in files:
        src_file = os.path.join(root, file)
        dst_file = os.path.join(dst_dir, file)
        shutil.copy(src_file, dst_file)
