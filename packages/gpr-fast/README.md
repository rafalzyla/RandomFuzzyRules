## Origin and attribution

This package is a modified and optimized implementation derived from
[gpr-algorithm](https://github.com/czmilanna/gpr-algorithm).

The original project was distributed under the MIT License. The original
copyright notice and license text are preserved in the `LICENSE` file in this
directory.

The modifications include:

- optimized rule-set evaluation;
- Numba-based computational kernels;
- added explicit random seeding during population initialization to improve
  reproducibility across repeated executions;
- renamed the main implementation module from `algorithm.py` to
  `classifier.py` to better reflect its role and align its naming with
  the scikit-learn estimator interface;
- renamed the main classifier class from `GPR` to `GPR_FAST` to distinguish
  the optimized implementation from the original version;

The modified implementation is used as the GPR reference classifier in the
scalability and final comparison experiments.