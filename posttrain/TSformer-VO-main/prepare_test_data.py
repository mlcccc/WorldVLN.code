import json
import random
import os
import shutil
from tqdm import tqdm

def main():
    jsonl_path = "/home/dataset-assist-0/xjc/train_p2p.jsonl"
    dest_root = "/home/dataset-assist-0/xjc/TSformer-VO-main/test_data_latent"
    num_samples = 199

    print(f"Reading {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total trajectories found: {total_lines}")

    if total_lines < num_samples:
        print(f"Warning: Only {total_lines} available, selecting all.")
        selected_lines = lines
    else:
        selected_lines = random.sample(lines, num_samples)
        print(f"Randomly selected {num_samples} trajectories.")

    success_count = 0
    skip_count = 0
    error_count = 0

    for line in tqdm(selected_lines):
        try:
            data = json.loads(line)
            source = data.get("source")
            traj_id = data.get("id")
            latent_path = data.get("latent_path")

            if not source or not traj_id or not latent_path:
                print(f"Skipping invalid line: {line.strip()}")
                skip_count += 1
                continue

            # Identify source directory (reshape_actionhead_data)
            src_dir = os.path.dirname(latent_path)
            if not os.path.basename(src_dir) == "reshape_actionhead_data":
                 # Just in case the structure is different, but based on example it should be this
                 # If latent_path is .../video_summed_codes.npy, dirname is the folder.
                 pass
            
            if not os.path.exists(src_dir):
                print(f"Source directory not found: {src_dir}")
                error_count += 1
                continue

            # Construct destination path
            # Structure: test_data_latent/{source}/{id}/reshape_actionhead_data
            dest_dir = os.path.join(dest_root, source, traj_id, "reshape_actionhead_data")

            if os.path.exists(dest_dir):
                # print(f"Destination exists, skipping: {dest_dir}")
                # We can skip or overwrite. Let's overwrite/merge to be safe
                pass

            # Copy
            # copytree with dirs_exist_ok=True requires Python 3.8+
            os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
            success_count += 1

        except Exception as e:
            print(f"Error processing line: {e}")
            error_count += 1

    print(f"Done. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
