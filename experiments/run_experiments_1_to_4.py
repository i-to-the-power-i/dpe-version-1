"""Run experiment scripts 1–4 concurrently and record wall-clock runtimes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import subprocess
import sys
import time

import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "experiment-execution-times-1-to-4.csv"
SCRIPTS = {
    1: "experiment-1.py",
    2: "experiment-2.py",
    3: "experiment-3.py",
    4: "experiment-4.py",
}


def run_one(number, script):
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, script],
            cwd=HERE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        status = "success" if completed.returncode == 0 else f"failed ({completed.returncode})"
        output_tail = completed.stdout[-4000:]
    except Exception as error:
        status = "failed (launch error)"
        output_tail = repr(error)
    elapsed = time.perf_counter() - start
    return {
        "experiment": number,
        "script": script,
        "started_at_utc": started_at.isoformat(timespec="seconds"),
        "execution_time_seconds": round(elapsed, 3),
        "status": status,
        "output_tail": output_tail,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', type=int, choices=sorted(SCRIPTS),
                        default=sorted(SCRIPTS), help='Experiment numbers to run concurrently')
    selected = {number: SCRIPTS[number] for number in parser.parse_args().experiments}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = [pool.submit(run_one, n, script) for n, script in selected.items()]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"Experiment {row['experiment']}: {row['status']} in {row['execution_time_seconds']:.3f}s")

    new_rows = pd.DataFrame(rows)
    if OUTPUT.exists():
        previous = pd.read_csv(OUTPUT)
        previous = previous[~previous.experiment.isin(new_rows.experiment)]
        new_rows = pd.concat([previous, new_rows], ignore_index=True)
    new_rows.sort_values("experiment").to_csv(OUTPUT, index=False)
    print(f"Runtime table saved to {OUTPUT}")


if __name__ == "__main__":
    main()
