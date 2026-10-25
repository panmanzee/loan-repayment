"""รันทุก notebook: python Model/run_all_notebooks.py [Regression|Classification|ชื่อไฟล์]"""

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(path):
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    try:
        ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": os.path.dirname(path)}}
        )
    except CellExecutionError as e:
        print(f"ERROR in {os.path.relpath(path, ROOT)}:\n{str(e)[-1500:]}", flush=True)
    nbformat.write(nb, path)
    errors = [
        o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"
    ]
    return os.path.relpath(path, ROOT), time.time() - t0, len(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="", help="Regression, Classification, or text in a notebook file name")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    for task in ("Regression", "Classification"):
        if args.filter in ("Regression", "Classification") and args.filter != task:
            continue
        books = sorted(glob.glob(os.path.join(ROOT, "Notebooks", task, "*.ipynb")))
        if args.filter not in ("", task):
            books = [p for p in books if args.filter in os.path.basename(p)]
        models = [p for p in books if "Model_Comparison" not in p]
        last = [p for p in books if "Model_Comparison" in p]
        for group in (models, last):
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for name, seconds, n_err in pool.map(run, group):
                    print(f"{'OK ' if n_err == 0 else 'ERR'} {name:<60} {seconds:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
