"""Persistent Python 3.10 worker used by GPRFastSubprocessClassifier."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback

import numpy as np
import numba


numba_threads = os.environ.get("NUMBA_NUM_THREADS")

if numba_threads is not None:
    numba.set_num_threads(int(numba_threads))

with contextlib.redirect_stdout(sys.stderr):
    from gpr_fast import GPR_FAST


def _send(payload):
    print(json.dumps(payload), flush=True)


def _load_array(path):
    return np.load(path, allow_pickle=False)


def main():
    model = None
    _send({"status": "ready"})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]

            if command == "shutdown":
                _send({"status": "ok"})
                return

            if command == "fit":
                X = _load_array(request["x_path"])
                y = _load_array(request["y_path"])
                model = GPR_FAST(**request["parameters"])
                start = time.perf_counter()
                # Protect the JSON protocol from estimator progress output.
                with contextlib.redirect_stdout(sys.stderr):
                    model.fit(X, y)
                elapsed = time.perf_counter() - start
            
                # Generate the textual model representation after stopping the fit timer.
                # The returned list contains one entry per learned rule followed by the
                # default ELSE rule.
                with contextlib.redirect_stdout(sys.stderr):
                    rules = list(model.rules)
            
                _send({"status": "ok", "elapsed_seconds": elapsed, "rules": rules})
            
                continue

            if model is None:
                raise RuntimeError("The worker has not fitted a model yet.")

            if command == "predict":
                X = _load_array(request["x_path"])
                start = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    output = np.asarray(model.predict(X))
                elapsed = time.perf_counter() - start
                np.save(request["output_path"], output, allow_pickle=False)
                _send({"status": "ok", "elapsed_seconds": elapsed})
                continue

            if command == "predict_proba":
                X = _load_array(request["x_path"])
                start = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    if hasattr(model, "predict_proba"):
                        output = np.asarray(model.predict_proba(X), dtype=np.float64)
                    else:
                        prediction = np.asarray(model.predict(X), dtype=np.float64)
                        output = np.column_stack((1.0 - prediction, prediction))
                elapsed = time.perf_counter() - start
                np.save(request["output_path"], output, allow_pickle=False)
                _send({"status": "ok", "elapsed_seconds": elapsed})
                continue

            raise ValueError(f"Unknown worker command: {command!r}")

        except Exception as error:
            _send(
                {
                    "status": "error",
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    main()
