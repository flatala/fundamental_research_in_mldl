#!/usr/bin/env python3

import argparse
import torch
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Check a PyTorch checkpoint and add a missing key with default value 0"
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the .pth checkpoint file to inspect and update"
    )
    parser.add_argument(
        "-k", "--key",
        default="epoch",
        help="Name of the key to check/add (default: 'epoch')"
    )
    args = parser.parse_args()

    try:
        data = torch.load(args.checkpoint, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print("Loaded object is not a dict; cannot check keys.")
        sys.exit(1)

    if args.key in data:
        print(f"Key '{args.key}' already present with value: {data[args.key]!r}")
    else:
        print(f"Key '{args.key}' not found. Adding with default value 0.")
        data[args.key] = 0
        try:
            torch.save(data, args.checkpoint)
            print(f"Updated checkpoint saved to {args.checkpoint}")
        except Exception as e:
            print(f"Error saving updated checkpoint: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
