import os
import numpy as np

def fix_key_in_npy_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'finetune.npy':
                filepath = os.path.join(dirpath, filename)
                try:
                    data = np.load(filepath, allow_pickle=True).item()
                    if 'times' in data and 'time' not in data:
                        data['time'] = data.pop('times')
                        np.save(filepath, data)
                        print(f"Fixed key in file: {filepath}")
                except Exception as e:
                    print(f"Failed to process file {filepath}: {e}")

# Call the function with the root directory containing all subdirectories
root_directory = 'C:/Ebooks/code/LlaMAft/results'
fix_key_in_npy_files(root_directory)