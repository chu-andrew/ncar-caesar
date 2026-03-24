import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).parent.parent


def discover_modules() -> list[str]:
    return sorted(
        str(p.relative_to(SRC)).replace("/", ".").removesuffix(".py")
        for p in SRC.rglob("plot_*.py")
    )


def run(module: str) -> tuple[str, int, str]:
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        cwd=SRC,
    )
    output = (result.stdout + result.stderr).strip()
    return module, result.returncode, output


def main():
    modules = discover_modules()
    print(f"Found {len(modules)} plot scripts: {', '.join(modules)}\n")

    with ThreadPoolExecutor(max_workers=len(modules)) as pool:
        futures = {pool.submit(run, m): m for m in modules}
        for future in as_completed(futures):
            module, returncode, output = future.result()
            status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
            print(f"=== {module} [{status}] ===")
            if output:
                print(output)
            print()


if __name__ == "__main__":
    main()
