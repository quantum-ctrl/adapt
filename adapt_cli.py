"""Command-line launcher for ADAPT."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Sequence


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MP_API_KEY = "sMqgB9CvxOwdio2tBv3XYZm3uDCYaH5c"


def _set_default_environment(host: str | None = None, port: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MP_API_KEY", DEFAULT_MP_API_KEY)
    if host is not None:
        env["ADAPT_HOST"] = host
    if port is not None:
        env["ADAPT_PORT"] = str(port)
    os.environ.update({key: value for key, value in env.items() if key in {"MP_API_KEY", "ADAPT_HOST", "ADAPT_PORT"}})
    return env


def _host_port(args: argparse.Namespace) -> tuple[str, int]:
    host = args.host or os.environ.get("ADAPT_HOST", DEFAULT_HOST)
    port = args.port or int(os.environ.get("ADAPT_PORT", DEFAULT_PORT))
    return host, port


def run_browser(initial_folder: str | None = None) -> int:
    _set_default_environment()
    from ADAPT_browser.app import main as browser_main

    return browser_main(initial_folder=initial_folder)


def run_edit(host: str, port: int, reload: bool = False) -> int:
    _set_default_environment(host, port)
    import uvicorn

    print(f"Starting ADAPT Edit on http://{host}:{port}")
    uvicorn.run("ADAPT_edit.server:app", host=host, port=port, reload=reload)
    return 0


def run_both(initial_folder: str | None, host: str, port: int) -> int:
    env = _set_default_environment(host, port)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "ADAPT_edit.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    print(f"Starting ADAPT Edit in background on http://{host}:{port}")
    edit_process = subprocess.Popen(command, env=env)
    try:
        time.sleep(2)
        if edit_process.poll() is not None:
            return edit_process.returncode or 1

        print("Starting ADAPT Browser")
        return run_browser(initial_folder)
    finally:
        if edit_process.poll() is None:
            print("Stopping ADAPT Edit")
            edit_process.terminate()
            try:
                edit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                edit_process.kill()
                edit_process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adapt",
        description="Launch ADAPT Browser and ADAPT Edit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    both = subparsers.add_parser("both", help="Start ADAPT Edit and ADAPT Browser.")
    both.add_argument("initial_folder", nargs="?", help="Initial folder to open in the browser.")
    both.add_argument("--host", help=f"ADAPT Edit host. Defaults to ADAPT_HOST or {DEFAULT_HOST}.")
    both.add_argument("--port", type=int, help=f"ADAPT Edit port. Defaults to ADAPT_PORT or {DEFAULT_PORT}.")
    both.set_defaults(func=_run_both_command)

    browser = subparsers.add_parser("browser", help="Start ADAPT Browser.")
    browser.add_argument("initial_folder", nargs="?", help="Initial folder to open.")
    browser.set_defaults(func=_run_browser_command)

    edit = subparsers.add_parser("edit", help="Start ADAPT Edit.")
    edit.add_argument("--host", help=f"Host to bind. Defaults to ADAPT_HOST or {DEFAULT_HOST}.")
    edit.add_argument("--port", type=int, help=f"Port to bind. Defaults to ADAPT_PORT or {DEFAULT_PORT}.")
    edit.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    edit.set_defaults(func=_run_edit_command)

    return parser


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"both", "browser", "edit"}
    help_flags = {"-h", "--help"}
    if not args:
        return ["both"]
    if args[0] in commands or args[0] in help_flags:
        return args
    return ["both", *args]


def _run_browser_command(args: argparse.Namespace) -> int:
    return run_browser(args.initial_folder)


def _run_edit_command(args: argparse.Namespace) -> int:
    host, port = _host_port(args)
    return run_edit(host, port, reload=args.reload)


def _run_both_command(args: argparse.Namespace) -> int:
    host, port = _host_port(args)
    return run_both(args.initial_folder, host, port)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
