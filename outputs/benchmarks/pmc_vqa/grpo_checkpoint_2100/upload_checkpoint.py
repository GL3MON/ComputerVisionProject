"""Upload the validated merged checkpoint to Hugging Face."""
from pathlib import Path
from datetime import datetime, timezone
import json
from huggingface_hub import HfApi

run = Path(__file__).resolve().parent
folder = run / "merged"
repo = "GL3MON/qwen3-vl-2b-medical-pmcvqa-grpo-merged"
api = HfApi()
who = api.whoami()
assert who["name"] == "GL3MON", "The authenticated account does not match the repository owner"
assert (folder / "model.safetensors").is_file()
assert (folder / "README.md").is_file()
base = api.model_info("GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged")
print("Creating or opening repository:", repo, flush=True)
url = api.create_repo(repo_id=repo, repo_type="model", private=base.private, exist_ok=True)
print("Uploading merged model, processor, model card, and evaluation metrics", flush=True)
commit = api.upload_folder(
    repo_id=repo, repo_type="model", folder_path=str(folder),
    ignore_patterns=[".cache/**", "*.tmp"],
    commit_message="Upload merged medical GRPO checkpoint-2100 and PMC-VQA evaluation",
)
info = api.model_info(repo, revision=commit.oid, files_metadata=True)
remote = {f.rfilename:f for f in info.siblings}
local = [f for f in folder.rglob("*") if f.is_file() and ".cache" not in f.relative_to(folder).parts]
for f in local:
    relative = str(f.relative_to(folder))
    assert relative in remote, "Missing uploaded file: " + relative
    assert remote[relative].size == f.stat().st_size, "Uploaded size differs: " + relative
metadata = {"repo_id":repo, "repo_url":str(url), "commit_url":commit.commit_url,
            "revision":commit.oid, "private":info.private,
            "checkpoint_step":2100, "base_model":base.id,
            "uploaded_at":datetime.now(timezone.utc).isoformat(),
            "verified_files":[str(f.relative_to(folder)) for f in local],
            "merged_vision_and_language_adapters":True}
(run / "hf_upload_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2), flush=True)
