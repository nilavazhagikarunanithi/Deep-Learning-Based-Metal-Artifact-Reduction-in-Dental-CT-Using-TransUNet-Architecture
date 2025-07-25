#!/usr/bin/env python3
"""
generate_absolute_masks.py

For each "difference" .npy file in your T1 difference directory, this script:
  1) Loads the absolute-difference map
  2) Creates a binary mask where mask=1 if difference>0, else 0
  3) Saves the mask as a .npy in your output folder

Usage:
  python generate_absolute_masks.py \
      --diff_dir "D:/ct data/t1 difference npy" \
      --mask_dir "D:/ct data/t1 absolute_mask npy"
"""

import os
import argparse
import glob
import numpy as np

def make_masks(diff_dir: str, mask_dir: str):
    # ensure output directory exists
    os.makedirs(mask_dir, exist_ok=True)

    # find all .npy files in the diff_dir
    diff_paths = sorted(glob.glob(os.path.join(diff_dir, '*.npy')))
    if not diff_paths:
        raise RuntimeError(f"No .npy files found in {diff_dir!r}")

    for diff_path in diff_paths:
        # load the difference map
        diff = np.load(diff_path)

        # generate binary mask: 1 where diff>0, else 0
        mask = (diff > 0).astype(np.uint8)

        # derive an output filename
        fname = os.path.basename(diff_path)          # e.g. "p-1_difference.npy"
        base, _ = os.path.splitext(fname)            # "p-1_difference"
        # replace "_difference" with "_abs_mask" (or just append)
        if base.endswith('_difference'):
            out_base = base[:-len('_difference')] + '_abs_mask'
        else:
            out_base = base + '_abs_mask'
        out_path = os.path.join(mask_dir, out_base + '.npy')

        # save mask
        np.save(out_path, mask)
        print(f"Saved mask: {out_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Generate binary masks from absolute-difference .npy files")
    p.add_argument('--diff_dir',  type=str, required=True,
                   help="Directory containing your *_difference.npy files")
    p.add_argument('--mask_dir',  type=str, required=True,
                   help="Output directory for *_abs_mask.npy files")
    args = p.parse_args()

    make_masks(args.diff_dir, args.mask_dir)
