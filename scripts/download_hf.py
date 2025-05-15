from huggingface_hub import snapshot_download

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the data from Hugging Face. Use --no-data to skip.",
    )
    parser.add_argument(
        "--models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the models from Hugging Face. Use --no-models to skip.",
    )
    args = parser.parse_args()

    if args.data:
        print("Downloading data from Hugging Face to data/...")
        snapshot_download(
            "ai4co/routefinder",
            allow_patterns=["data/*"],
            local_dir=".",
            repo_type="dataset",
        )

    if args.models:
        print("Downloading models from Hugging Face to checkpoints/...")
        snapshot_download(
            "ai4co/routefinder", allow_patterns=["checkpoints/*"], local_dir="."
        )

    if not args.data and not args.models:
        print("No download requested. Exiting.")
        exit()
