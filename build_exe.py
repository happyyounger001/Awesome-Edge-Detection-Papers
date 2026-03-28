from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_pyinstaller_command(onefile: bool = False) -> list[str]:
    separator = ';' if sys.platform.startswith('win') else ':'
    command = [
        sys.executable,
        '-m',
        'PyInstaller',
        '--noconfirm',
        '--clean',
        '--name',
        'FencingLungeAITrainer',
        '--windowed',
        '--add-data',
        f'{ROOT / "config.yaml"}{separator}.',
        '--collect-all',
        'mediapipe',
        '--collect-all',
        'matplotlib',
        '--collect-all',
        'pandas',
        str(ROOT / 'main.py'),
    ]
    if onefile:
        command.insert(5, '--onefile')
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description='Build Windows EXE with PyInstaller')
    parser.add_argument('--onefile', action='store_true', help='Build a single-file EXE instead of one-folder output')
    args = parser.parse_args()

    command = build_pyinstaller_command(onefile=args.onefile)
    print('Running:', ' '.join(str(part) for part in command))
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
