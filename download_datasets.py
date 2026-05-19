#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

TOP_LEVEL_COPY = [
    "datasets",
    "README.md"
    "README",
    "LICENSE",
    "LICENSE.md",
    ".gitattributes",
]

CRUXEVAL_HF_REPO = "cruxeval-org/cruxeval"
CRUXEVAL_HF_SPLIT = "test"

def safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """Safely extract tar.gz into dest_dir (prevents path traversal)."""
    dest_dir = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            member_path = (dest_dir / m.name).resolve()
            if not str(member_path).startswith(str(dest_dir) + os.sep):
                raise RuntimeError(f"Unsafe tar member path: {m.name}")
        tf.extractall(dest_dir)

def patterns_for_selected(averitec: bool, tabfact: bool, ricechem: bool):
    pats = []
    if averitec:
        pats.append("datasets/AVeriTeC/**")
    if tabfact:
        pats.append("datasets/TabFact/**")
    if ricechem:
        pats.append("datasets/RiceChem/**")
    pats.extend(["README*", "LICENSE*", ".gitattributes"])
    return pats


def need_hf_snapshot(sel: dict) -> bool:
    """True if any dataset hosted in the project's HF data repo is selected."""
    return any(sel[k] for k in ("averitec", "tabfact", "ricechem"))

def download_cruxeval(statics_dir: Path) -> None:
    """Fetch CRUXEval from HuggingFace and write it as test.jsonl in the
    layout expected by CRUXEvalDataset: statics/datasets/CRUXEval/test.jsonl."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "CRUXEval download requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from e

    dest_dir = statics_dir / "datasets" / "CRUXEval"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "test.jsonl"

    hf = load_dataset(CRUXEVAL_HF_REPO, split=CRUXEVAL_HF_SPLIT)
    with open(dest_path, "w", encoding="utf-8") as f:
        for row in hf:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    print(f"CRUXEval: wrote {len(hf)} samples to {dest_path}")


def copy_into(src_root: Path, dst_root: Path, rel_path: str) -> None:
    src = src_root / rel_path
    if not src.exists():
        return
    dst = dst_root / rel_path

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.rglob("*"):
            rel = p.relative_to(src)
            out = dst / rel
            if p.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def resolve_project_path(cli_project_path: str | None) -> Path:
    if cli_project_path:
        return Path(cli_project_path).expanduser().resolve()
    env = os.environ.get("PROJECT_PATH")
    if not env:
        raise SystemExit(
            "PROJECT_PATH is not set. Please either:\n"
            "  1) export PROJECT_PATH=/path/to/project\n"
            "  or\n"
            "  2) pass --project_path /path/to/project\n"
        )
    return Path(env).expanduser().resolve()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default=None,
                    help="HF dataset repo id for AVeriTeC/TabFact/RiceChem, "
                         "e.g. THunderCondOR/breaking-the-chain-intervention-data. "
                         "Required unless only --only cruxeval is requested.")
    ap.add_argument("--project_path", default=None, help="Overrides PROJECT_PATH env var (optional)")

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Download all datasets (default)")
    group.add_argument("--only", nargs="+", choices=["averitec", "tabfact", "ricechem", "cruxeval"],
                       help="Download only selected datasets")

    ap.add_argument("--no_extract", action="store_true", help="Do not extract TabFact all_csv.tar.gz")
    args = ap.parse_args()

    project_path = resolve_project_path(args.project_path)
    statics_dir = project_path / "statics"
    statics_dir.mkdir(parents=True, exist_ok=True)

    all_keys = ["averitec", "tabfact", "ricechem", "cruxeval"]
    if args.all or args.only is None:
        sel = {k: True for k in all_keys}
    else:
        sel = {k: False for k in all_keys}
        for k in args.only:
            sel[k] = True

    if need_hf_snapshot(sel):
        if not args.repo_id:
            raise SystemExit(
                "--repo_id is required when downloading AVeriTeC/TabFact/RiceChem."
            )

        allow_patterns = patterns_for_selected(
            averitec=sel["averitec"],
            tabfact=sel["tabfact"],
            ricechem=sel["ricechem"],
        )

        tmp_dir = statics_dir / "_hf_tmp_download"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(tmp_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )

        for item in TOP_LEVEL_COPY:
            copy_into(tmp_dir, statics_dir, item)

        if sel["tabfact"] and (not args.no_extract):
            tar_path = statics_dir / "datasets" / "TabFact" / "data" / "all_csv.tar.gz"
            if tar_path.exists():
                dest = tar_path.parent  # .../datasets/TabFact/data
                all_csv_dir = dest / "all_csv"
                if not all_csv_dir.exists():
                    safe_extract_tar(tar_path, dest)
            else:
                print(f"Not found: {tar_path} (did you upload the archive?)")

        # Cleanup temp folder (removes any .cache created during download)
        shutil.rmtree(tmp_dir)

    if sel["cruxeval"]:
        download_cruxeval(statics_dir)

    print(f"Done. statics is ready at: {statics_dir}")

if __name__ == "__main__":
    main()
