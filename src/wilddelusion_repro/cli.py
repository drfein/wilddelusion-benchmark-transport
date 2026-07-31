"""Command-line interface for the complete reproduction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

from .integrity import verify_packaged_results
from .pipeline import Pipeline


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_pipeline(args: argparse.Namespace) -> Pipeline:
    root = repository_root()
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    return Pipeline.load(root, config)


def command_list(args: argparse.Namespace) -> None:
    pipeline = load_pipeline(args)
    for name, stage in pipeline.stages.items():
        deps = ", ".join(stage.deps) or "-"
        print(f"{name:30} {stage.kind:7} deps={deps}\n  {stage.description}")


def command_doctor(args: argparse.Namespace) -> None:
    pipeline = load_pipeline(args)
    packages = ["numpy", "pandas", "scipy", "openai", "datasets", "tqdm"]
    report = {
        "repository": str(pipeline.root),
        "config": str(pipeline.config_path),
        "python_packages": {
            package: importlib.util.find_spec(package) is not None
            for package in packages
        },
        "executables": {
            name: shutil.which(name) for name in ["git", "nvidia-smi"]
        },
        "environment": {
            name: bool(os.getenv(name))
            for name in ["OPENAI_API_KEY", "HF_TOKEN", "WD_NATURAL_CONTROL_DIR"]
        },
    }
    print(json.dumps(report, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    root = repository_root()
    report = verify_packaged_results(root)
    exact = (
        root
        / "experiments"
        / "lost_in_delusion_replication_260600975"
        / "cross_domain_comparison"
        / "verify_abstract_claims.py"
    )
    subprocess.run([os.sys.executable, str(exact)], cwd=exact.parents[1], check=True)
    print(json.dumps(report, indent=2))


def command_run(args: argparse.Namespace) -> None:
    pipeline = load_pipeline(args)
    pipeline.run(
        args.targets,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        allow_api=args.allow_api,
        allow_gpu=args.allow_gpu,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", default="configs/pipeline.toml")
    commands = root.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List auditable stages")
    list_parser.set_defaults(func=command_list)

    doctor = commands.add_parser("doctor", help="Check local prerequisites")
    doctor.set_defaults(func=command_doctor)

    verify = commands.add_parser("verify", help="Verify packaged result gates")
    verify.set_defaults(func=command_verify)

    run = commands.add_parser("run", help="Run stages and their dependencies")
    run.add_argument("targets", nargs="+", default=["all"])
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--allow-api", action="store_true")
    run.add_argument("--allow-gpu", action="store_true")
    run.set_defaults(func=command_run)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
