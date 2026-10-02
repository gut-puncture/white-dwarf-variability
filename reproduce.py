#!/usr/bin/env python3
"""Download the recorded observations and rebuild the two-star analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import venv

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_records(root, records):
    for record in records:
        path = root / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Missing or changed input: {record['path']}")


def unpack(archive, destination, records):
    # Only regular files listed in the manifest may leave this archive.
    expected = {r["path"]: r for r in records}
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if member.name not in expected or not member.isfile() or member.name in seen:
                raise RuntimeError(f"Unexpected archive member: {member.name}")
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Invalid archive path: {member.name}")
            if member.size != expected[member.name]["bytes"]:
                raise RuntimeError(f"Wrong file size: {member.name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            seen.add(member.name)
    if seen != set(expected):
        raise RuntimeError("The archive is missing observations.")
    verify_records(destination, records)


def get_observations(override):
    manifest = json.loads((ROOT / "provenance/raw_manifest.json").read_text())
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    archive = Path(override).resolve() if override else cache / "raw-observations-v1.tar.gz"
    if not archive.exists():
        if override:
            raise FileNotFoundError(archive)
        info = manifest["archive"]
        print(f"Downloading {info['bytes'] / 1e6:.0f} MB of original observations…", flush=True)
        partial = archive.with_suffix(".part")
        request = urllib.request.Request(info["url"], headers={"User-Agent": "white-dwarf-variability/1.0"})
        with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as output:
            total = 0
            last_report = time.monotonic()
            while block := source.read(1024 * 1024):
                output.write(block)
                total += len(block)
                if time.monotonic() - last_report > 10:
                    print(f"  {total / 1e6:.0f} MB downloaded", flush=True)
                    last_report = time.monotonic()
        if sha256(partial) != info["sha256"]:
            raise RuntimeError("Download checksum differs from the recorded observations. Nothing was analysed.")
        partial.replace(archive)
    if sha256(archive) != manifest["archive"]["sha256"]:
        raise RuntimeError("Archive checksum mismatch. Nothing was analysed.")
    destination = cache / "observations"
    if not destination.exists():
        temporary = cache / "observations-unpacking"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        unpack(archive, temporary, manifest["files"])
        temporary.rename(destination)
    verify_records(destination, manifest["files"])
    print(f"Verified {len(manifest['files'])} original files.", flush=True)
    return destination


def bootstrap(args):
    if sys.version_info < (3, 11):
        raise RuntimeError("Install Python 3.11 or newer, then run this command again.")
    if args.use_current_python:
        return
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating a separate Python environment…", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    lock = sha256(ROOT / "requirements.txt")
    stamp = environment / ".requirements-sha256"
    if not stamp.exists() or stamp.read_text().strip() != lock:
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")], check=True)
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        stamp.write_text(lock + "\n")
    command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:], "--use-current-python"]
    raise SystemExit(subprocess.run(command).returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-archive", help="Use an already downloaded archive (its checksum must match).")
    parser.add_argument("--output", default="results", help="New output directory; existing results are never overwritten.")
    parser.add_argument("--use-current-python", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"{output} already exists. Keep it and choose a new name: --output results-2")
    bootstrap(args)
    verify_records(ROOT, json.loads((ROOT / "provenance/input_checksums.json").read_text()))
    observations = get_observations(args.data_archive)
    output.mkdir(parents=True)
    for name in ("scripts", "inputs/data"):
        shutil.copytree(ROOT / name, output / ("scripts" if name == "scripts" else "data"))
    shutil.copytree(observations / "raw", output / "raw")
    (output / "figures").mkdir()
    (output / "logs").mkdir()
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    started = time.monotonic()
    status = subprocess.run([sys.executable, str(ROOT / "run_analysis.py"), str(output)], env=environment).returncode
    if status:
        raise RuntimeError(f"Analysis stopped. See {output / 'logs'}. No successful reproduction is claimed.")
    subprocess.run([sys.executable, str(ROOT / "verify_results.py"), str(output)], env=environment, check=True)
    print(f"\nFinished in {(time.monotonic() - started) / 60:.1f} minutes. Read {output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"\nSTOPPED: {error}", file=sys.stderr)
        raise SystemExit(1)
