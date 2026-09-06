"""Scikit-learn-compatible bridge to GPR_FAST running under Python 3.10.

The wrapper itself runs in the main Python 3.12 environment. A persistent
worker process is started through ``uv`` in the dedicated GPR environment.
Training and prediction arrays are exchanged through temporary ``.npy`` files,
while commands and status messages use a small JSON-lines protocol.

The fitted GEPPY/DEAP estimator remains in memory inside the Python 3.10
worker. It is neither pickled nor imported into Python 3.12.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GPRFastSubprocessClassifier(ClassifierMixin, BaseEstimator):
    """Run GPR_FAST in a persistent Python 3.10 subprocess.

    Parameters mirror the current GPR_FAST constructor. The wrapper follows
    the scikit-learn estimator interface sufficiently for sequential fitting,
    prediction, cloning, and the project's benchmark runner.

    Parameters
    ----------
    n_populations : int, default=100
        Population size used by GPR_FAST.

    n_generations : int, default=100
        Number of evolutionary generations.

    threshold : float, default=0.5
        Positive-class decision threshold.

    verbose : bool, default=False
        Verbosity passed to GPR_FAST. Worker stdout is redirected to stderr so
        it cannot corrupt the JSON communication protocol.

    max_n_of_rules : int, default=6
        Maximum number of rules in an individual.

    max_n_of_ands : int, default=6
        Maximum number of conjunctions in a rule.

    base_pb : float, default=0.1
        Base genetic-operator probability.

    random_state : int or None, default=42
        Random seed passed to GPR_FAST.

    feature_names : sequence of str or None, default=None
        Optional transformed feature names. If omitted, ``x1``, ``x2``, ...
        are created during fitting.

    gpr_project : str, pathlib.Path, or None, default=None
        Path to the dedicated uv project. The default is
        ``<repository>/environments/gpr``.

    uv_executable : str, default="uv"
        Name or path of the uv executable.

    worker_module : str, default="experiments.gpr_fast_worker"
        Python module executed in the worker environment.

    n_jobs : int, default=None
        Number of CPU cores used during computations.

    Attributes
    ----------
    classes_ : ndarray
        Classes observed during fitting.

    n_features_in_ : int
        Number of fitted input features.

    rules_ : list of str
        Textual representation of the fitted GPR rule set returned by the
        Python 3.10 worker. The list contains one ``IF ... THEN ...`` entry for
        every learned positive-class rule, followed by the default ``ELSE`` rule.

    worker_fit_time_ : float
        Time spent inside ``GPR_FAST.fit`` in the worker. The measurement excludes
        worker startup, file transfer, JSON communication, and generation of the
        textual ``rules_`` representation.

    subprocess_fit_wall_time_ : float
        Total wrapper fit time, including worker startup, array transfer,
        communication, and extraction of the textual rule representation.

    Notes
    -----
    The fitted estimator owns a live subprocess and is not serializable.
    Sequential cross-validation is supported. Do not send a fitted instance
    to another process or use it with process-based parallel prediction.
    """

    def __init__(
        self,
        n_populations=100,
        n_generations=100,
        threshold=0.5,
        verbose=False,
        max_n_of_rules=6,
        max_n_of_ands=6,
        base_pb=0.1,
        random_state=42,
        feature_names=None,
        gpr_project=None,
        uv_executable="uv",
        worker_module="experiments.gpr_fast_worker",
        n_jobs=None
    ):
        self.n_populations = n_populations
        self.n_generations = n_generations
        self.threshold = threshold
        self.verbose = verbose
        self.max_n_of_rules = max_n_of_rules
        self.max_n_of_ands = max_n_of_ands
        self.base_pb = base_pb
        self.random_state = random_state
        self.feature_names = feature_names
        self.gpr_project = gpr_project
        self.uv_executable = uv_executable
        self.worker_module = worker_module
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit GPR_FAST in the persistent Python 3.10 worker."""
        X, y = check_X_y(X, y, dtype=np.float64, force_all_finite=True)
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y)

        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("GPRFastSubprocessClassifier supports binary classification only.")

        self.close()

        if hasattr(self, "rules_"):
            del self.rules_
        
        wall_start = time.perf_counter()
        self._start_worker()

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        feature_names = (
            list(self.feature_names)
            if self.feature_names is not None
            else [f"x{i + 1}" for i in range(X.shape[1])]
        )
        if len(feature_names) != X.shape[1]:
            self.close()
            raise ValueError("feature_names must match the number of input columns.")

        x_path = self._work_dir / "fit_X.npy"
        y_path = self._work_dir / "fit_y.npy"
        np.save(x_path, X, allow_pickle=False)
        np.save(y_path, y, allow_pickle=False)

        response = self._request(
            {
                "command": "fit",
                "x_path": str(x_path),
                "y_path": str(y_path),
                "parameters": {
                    "feature_names": feature_names,
                    "n_populations": self.n_populations,
                    "n_generations": self.n_generations,
                    "threshold": self.threshold,
                    "verbose": self.verbose,
                    "max_n_of_rules": self.max_n_of_rules,
                    "max_n_of_ands": self.max_n_of_ands,
                    "base_pb": self.base_pb,
                    "random_state": self.random_state,
                },
            }
        )

        if "rules" not in response:
            self.close()
            raise RuntimeError(
                "The GPR worker fit response did not contain "
                "the fitted rule representation."
            )
            
        self.worker_fit_time_ = float(response["elapsed_seconds"])
        self.subprocess_fit_wall_time_ = time.perf_counter() - wall_start

        # Store the textual rules returned by GPR_FAST. The list contains the
        # learned IF-THEN rules followed by the default ELSE rule.
        self.rules_ = list(response["rules"])
        
        # Optional hook used by the benchmark runner to exclude bridge overhead.
        self._benchmark_fit_time_ = self.worker_fit_time_
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict classes through the fitted Python 3.10 worker."""
        return self._predict_command(X, "predict")

    def predict_proba(self, X):
        """Return class probabilities through the Python 3.10 worker.

        If GPR_FAST does not implement ``predict_proba``, the worker returns a
        two-column hard-probability matrix derived from ``predict``. This keeps
        compatibility with the existing benchmark score fallback.
        """
        return self._predict_command(X, "predict_proba")

    def close(self):
        """Terminate the worker process and remove temporary files."""
        process = getattr(self, "_process", None)
        if process is not None:
            try:
                if process.poll() is None:
                    self._request({"command": "shutdown"}, allow_shutdown=True)
                    process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            finally:
                self._process = None

        temporary_directory = getattr(self, "_temporary_directory", None)
        if temporary_directory is not None:
            try:
                temporary_directory.cleanup()
            except Exception:
                pass
            self._temporary_directory = None
        self._work_dir = None

    def _predict_command(self, X, command):
        check_is_fitted(self, ["_is_fitted", "classes_", "n_features_in_"])
        X = check_array(X, dtype=np.float64, force_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the fitted estimator expects "
                f"{self.n_features_in_}."
            )
        X = np.ascontiguousarray(X, dtype=np.float64)

        token = f"{command}_{time.time_ns()}"
        x_path = self._work_dir / f"{token}_X.npy"
        output_path = self._work_dir / f"{token}_output.npy"
        np.save(x_path, X, allow_pickle=False)

        wall_start = time.perf_counter()
        response = self._request(
            {
                "command": command,
                "x_path": str(x_path),
                "output_path": str(output_path),
            }
        )
        output = np.load(output_path, allow_pickle=False)
        worker_elapsed = float(response["elapsed_seconds"])
        wall_elapsed = time.perf_counter() - wall_start
        self.worker_predict_time_ = worker_elapsed
        self.subprocess_predict_wall_time_ = wall_elapsed
        self._benchmark_predict_time_ = worker_elapsed

        for path in (x_path, output_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return output

    def _start_worker(self):
        worker_environment = os.environ.copy()

        if self.n_jobs is not None:
            n_jobs = int(self.n_jobs)
        
            if n_jobs == -1:
                for variable in (
                    "NUMBA_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    worker_environment.pop(variable, None)
        
            elif n_jobs >= 1:
                thread_count = str(n_jobs)
        
                for variable in (
                    "NUMBA_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    worker_environment[variable] = thread_count
        
            else:
                raise ValueError("n_jobs must be None, -1, or a positive integer.")
                
        repository_root = Path(__file__).resolve().parents[1]
        project = (
            Path(self.gpr_project).resolve()
            if self.gpr_project is not None
            else repository_root / "environments" / "gpr"
        )
        if not project.exists():
            raise FileNotFoundError(f"GPR uv project was not found: {project}")

        uv = shutil.which(self.uv_executable)
        if uv is None:
            raise FileNotFoundError(
                f"uv executable was not found: {self.uv_executable!r}"
            )

        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="gpr_fast_bridge_"
        )
        self._work_dir = Path(self._temporary_directory.name)

        command = [
            uv,
            "run",
            "--project",
            str(project),
            "--locked",
            "python",
            "-u",
            "-m",
            self.worker_module,
        ]
        self._process = subprocess.Popen(
            command,
            cwd=repository_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=worker_environment,
        )
        response = self._read_response()
        if response.get("status") != "ready":
            self.close()
            raise RuntimeError(f"GPR worker failed to initialize: {response}")

    def _request(self, payload, allow_shutdown=False):
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError(self._worker_failure_message("GPR worker is not running."))
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if response.get("status") == "error":
            raise RuntimeError(response.get("traceback", response.get("message", "Worker error")))
        if not allow_shutdown and response.get("status") != "ok":
            raise RuntimeError(f"Unexpected worker response: {response}")
        return response

    def _read_response(self):
        """Read the next JSON response, ignoring diagnostic stdout lines."""
        assert self._process.stdout is not None
    
        diagnostic_lines = []
    
        while True:
            line = self._process.stdout.readline()
    
            if not line:
                diagnostic_text = "".join(diagnostic_lines)
    
                message = "GPR worker terminated unexpectedly."
    
                if diagnostic_text:
                    message += "\nNon-JSON worker output:\n" + diagnostic_text
    
                raise RuntimeError(self._worker_failure_message(message))
    
            try:
                return json.loads(line)
    
            except json.JSONDecodeError:
                diagnostic_lines.append(line)
    
                # The worker protocol reserves stdout for JSON, but some
                # third-party libraries print optional-dependency warnings
                # directly to stdout. Ignore those lines and keep waiting
                # for the actual JSON response.

    def _worker_failure_message(self, message):
        process = getattr(self, "_process", None)
        stderr = ""
        if process is not None and process.poll() is not None and process.stderr is not None:
            stderr = process.stderr.read()
        return f"{message}\nWorker stderr:\n{stderr}"

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
