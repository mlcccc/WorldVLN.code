#!/usr/bin/env python3
"""Download flan-t5-xl text encoder from ModelScope (fallback for HuggingFace SSL issues)."""

import os
import shutil
import sys


def download_from_modelscope(save_path):
    """Download flan-t5-xl from ModelScope."""
    from modelscope.hub.snapshot_download import snapshot_download

    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading flan-t5-xl to {save_path}...")

    try:
        # Download to a temporary location first
        tmp_path = save_path + "_tmp"
        model_dir = snapshot_download("google/flan-t5-xl", cache_dir=tmp_path)
        print(f"Downloaded to: {model_dir}")

        # Move files to the correct location
        src = os.path.join(tmp_path, "google", "flan-t5-xl")
        if os.path.exists(src):
            for item in os.listdir(src):
                src_item = os.path.join(src, item)
                dst_item = os.path.join(save_path, item)
                if os.path.exists(dst_item):
                    if os.path.isdir(dst_item):
                        shutil.rmtree(dst_item)
                    else:
                        os.remove(dst_item)
                shutil.move(src_item, dst_item)
            shutil.rmtree(tmp_path, ignore_errors=True)
            print(f"Files moved to {save_path}")
        else:
            print(f"Warning: expected directory not found at {src}")
            print(f"Files may be at: {model_dir}")

        return True
    except Exception as e:
        print(f"ModelScope download failed: {e}")
        return False


def download_from_huggingface(save_path):
    """Download flan-t5-xl from HuggingFace."""
    from transformers import T5EncoderModel, T5Tokenizer

    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading flan-t5-xl from HuggingFace to {save_path}...")

    try:
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", legacy=True)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer saved.")

        model = T5EncoderModel.from_pretrained("google/flan-t5-xl", torch_dtype="auto")
        model.save_pretrained(save_path)
        print("Model saved.")

        return True
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return False


def main():
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "train", "checkpoints", "text_encoder", "flan-t5-xl-official",
    )

    # Check if already downloaded
    if os.path.exists(os.path.join(save_path, "config.json")):
        print(f"T5 already downloaded to {save_path}")
        return

    # Try HuggingFace first, then ModelScope
    if not download_from_huggingface(save_path):
        print("Trying ModelScope...")
        if not download_from_modelscope(save_path):
            print("Failed to download from both sources.")
            sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
