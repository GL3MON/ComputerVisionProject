"""Explicit uploads; importing this module never contacts Hugging Face."""

import argparse
from medical_vlm.core.paths import resolve_path
from medical_vlm.core.io import write_json


def upload_model(folder, repo_id, private=False):
    from huggingface_hub import HfApi

    folder = resolve_path(folder)
    if not (folder / "config.json").is_file() or not list(folder.glob("*.safetensors")):
        raise ValueError("Upload folder must contain a merged model config and safetensors weights")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="model",
        ignore_patterns=[".cache/**", "*.tmp"],
        commit_message="Upload merged model and evaluation artifacts",
    )
    metadata = {
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "revision": commit.oid,
        "commit_url": commit.commit_url,
    }
    write_json(folder.parent / "hf_upload_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    print(upload_model(args.folder, args.repo, args.private))


if __name__ == "__main__":
    main()
