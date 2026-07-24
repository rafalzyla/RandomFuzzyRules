"""RandomFuzzyRulesClassifier: standalone random-search fuzzy rule classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import time
import warnings

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

_STATE_TO_CODE = {"high": 0, "low": 1, "medium": 2, "present": 3, "absent": 4}


@dataclass(frozen=True, order=True)
class Condition:
    feature: int
    state: str
    modifier: int = 1

    def __post_init__(self):
        if self.state not in _STATE_TO_CODE:
            raise ValueError(f"Unknown state: {self.state!r}")
        if not 1 <= self.modifier <= 3:
            raise ValueError("modifier must be in [1, 3]")
        if self.state in {"medium", "present", "absent"} and self.modifier != 1:
            raise ValueError(f"{self.state!r} only supports modifier=1")


Rule = Tuple[Condition, ...]
RuleSet = Tuple[Rule, ...]


@dataclass(frozen=True)
class RuleStats:
    support: float
    lift: float
    n_covered: int
    n_positive_covered: int
    n_negative_covered: int


@njit(cache=True, fastmath=False, parallel=False)
def _score_rules_fast(X, features, states, modifiers, lengths):
    out = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        total = 0.0
        for r in range(lengths.shape[0]):
            activation = 1.0
            for k in range(lengths[r]):
                j = features[r, k]
                state = states[r, k]
                modifier = modifiers[r, k]
                x = X[i, j]
                if state == 0:
                    value = x if modifier == 1 else x * x if modifier == 2 else x * x * x
                elif state == 1:
                    low = 1.0 - x
                    value = low if modifier == 1 else low * low if modifier == 2 else low * low * low
                elif state == 2:
                    value = x * (1.0 - x)
                elif state == 3:
                    value = x
                else:
                    value = 1.0 - x
                activation *= value
            total += activation
        out[i] = total
    return out


@njit(cache=True, fastmath=False, parallel=True)
def _batch_accuracy(X, y, features, states, modifiers, lengths, n_rules, threshold):
    accuracies = np.empty(features.shape[0], dtype=np.float64)
    log_quarter = 2.0 * np.log(0.5)
    for c in prange(features.shape[0]):
        correct = 0
        for i in range(X.shape[0]):
            score = 0.0
            for r in range(n_rules[c]):
                activation = 1.0
                for k in range(lengths[c, r]):
                    j = features[c, r, k]
                    state = states[c, r, k]
                    modifier = modifiers[c, r, k]
                    x = X[i, j]
                    if state == 0:
                        value = x if modifier == 1 else x * x if modifier == 2 else x * x * x
                    elif state == 1:
                        low = 1.0 - x
                        value = low if modifier == 1 else low * low if modifier == 2 else low * low * low
                    elif state == 2:
                        value = x * (1.0 - x)
                    elif state == 3:
                        value = x
                    else:
                        value = 1.0 - x
                    activation *= value
                score += activation
            p = 1.0 - np.exp(log_quarter * score)
            correct += int((p >= threshold) == y[i])
        accuracies[c] = correct / X.shape[0]
    return accuracies


@njit(cache=True, fastmath=False, parallel=True)
def _batch_accuracy_threshold_half(X, y, features, states, modifiers, lengths, n_rules):
    accuracies = np.empty(features.shape[0], dtype=np.float64)
    for c in prange(features.shape[0]):
        correct = 0
        for i in range(X.shape[0]):
            score = 0.0
            for r in range(n_rules[c]):
                activation = 1.0
                for k in range(lengths[c, r]):
                    j = features[c, r, k]
                    state = states[c, r, k]
                    modifier = modifiers[c, r, k]
                    x = X[i, j]
                    if state == 0:
                        value = x if modifier == 1 else x * x if modifier == 2 else x * x * x
                    elif state == 1:
                        low = 1.0 - x
                        value = low if modifier == 1 else low * low if modifier == 2 else low * low * low
                    elif state == 2:
                        value = x * (1.0 - x)
                    elif state == 3:
                        value = x
                    else:
                        value = 1.0 - x
                    activation *= value
                score += activation
                if score >= 0.5:
                    break
            correct += int((score >= 0.5) == y[i])
        accuracies[c] = correct / X.shape[0]
    return accuracies


@njit(cache=True, fastmath=False, parallel=False)
def _generate_raw_candidates_fast(
    seed, n_raw, n_rules, max_rule_len, max_modifier, continuous_mask
):
    """Generate fixed-shape raw candidates in Numba; Python canonicalizes them."""
    np.random.seed(seed)
    n_features = continuous_mask.shape[0]
    features = np.full((n_raw, n_rules, max_rule_len), -1, dtype=np.int64)
    states = np.zeros((n_raw, n_rules, max_rule_len), dtype=np.uint8)
    modifiers = np.ones((n_raw, n_rules, max_rule_len), dtype=np.uint8)
    lengths = np.empty((n_raw, n_rules), dtype=np.int64)

    for c in range(n_raw):
        for r in range(n_rules):
            length = np.random.randint(1, max_rule_len + 1)
            lengths[c, r] = length
            for k in range(length):
                j = np.random.randint(0, n_features)
                features[c, r, k] = j
                if continuous_mask[j]:
                    state = np.random.randint(0, 3)  # high, low, medium
                    states[c, r, k] = state
                    if state < 2:
                        modifiers[c, r, k] = np.random.randint(1, max_modifier + 1)
                else:
                    states[c, r, k] = np.random.randint(3, 5)  # present, absent
    return features, states, modifiers, lengths


@njit(cache=True, fastmath=False, parallel=False)
def _canonicalize_raw_candidates_fast(features, states, modifiers, lengths, n_rules, group_ids):
    """Canonicalize raw candidates in-place and mark structurally valid rows.

    To preserve the sampled rule length exactly, candidates requiring duplicate
    removal or categorical redundancy removal are rejected rather than shortened.
    """
    n_candidates, max_rules, max_len = features.shape
    valid = np.ones(n_candidates, dtype=np.bool_)

    for c in range(n_candidates):
        # Validate and sort conditions within each rule.
        for r in range(n_rules):
            length = lengths[c, r]
            for a in range(length):
                fa = features[c, r, a]
                sa = states[c, r, a]
                ga = group_ids[fa]
                for b in range(a):
                    fb = features[c, r, b]
                    sb = states[c, r, b]
                    # One semantic condition per transformed feature.
                    if fa == fb:
                        valid[c] = False
                    # Within one-hot groups, at most one present condition is
                    # possible; present plus any other condition is redundant or
                    # contradictory and is rejected to preserve sampled length.
                    gb = group_ids[fb]
                    if ga >= 0 and ga == gb and (sa == 3 or sb == 3):
                        valid[c] = False

            if not valid[c]:
                break

            # Insertion sort conditions by feature, state, modifier.
            for a in range(1, length):
                f0 = features[c, r, a]
                s0 = states[c, r, a]
                m0 = modifiers[c, r, a]
                b = a - 1
                while b >= 0:
                    fb = features[c, r, b]
                    sb = states[c, r, b]
                    mb = modifiers[c, r, b]
                    greater = fb > f0 or (fb == f0 and (sb > s0 or (sb == s0 and mb > m0)))
                    if not greater:
                        break
                    features[c, r, b + 1] = fb
                    states[c, r, b + 1] = sb
                    modifiers[c, r, b + 1] = mb
                    b -= 1
                features[c, r, b + 1] = f0
                states[c, r, b + 1] = s0
                modifiers[c, r, b + 1] = m0

        if not valid[c]:
            continue

        # Sort rules by length, then lexicographically by conditions.
        for a in range(1, n_rules):
            lf = lengths[c, a]
            tf = features[c, a].copy()
            ts = states[c, a].copy()
            tm = modifiers[c, a].copy()
            b = a - 1
            while b >= 0:
                lb = lengths[c, b]
                greater = lb > lf
                if lb == lf:
                    greater = False
                    for k in range(lf):
                        if features[c, b, k] != tf[k]:
                            greater = features[c, b, k] > tf[k]
                            break
                        if states[c, b, k] != ts[k]:
                            greater = states[c, b, k] > ts[k]
                            break
                        if modifiers[c, b, k] != tm[k]:
                            greater = modifiers[c, b, k] > tm[k]
                            break
                if not greater:
                    break
                lengths[c, b + 1] = lengths[c, b]
                features[c, b + 1] = features[c, b]
                states[c, b + 1] = states[c, b]
                modifiers[c, b + 1] = modifiers[c, b]
                b -= 1
            lengths[c, b + 1] = lf
            features[c, b + 1] = tf
            states[c, b + 1] = ts
            modifiers[c, b + 1] = tm

        # Reject duplicate rules because canonicalization would reduce n_rules.
        for r in range(1, n_rules):
            if lengths[c, r] == lengths[c, r - 1]:
                same = True
                for k in range(lengths[c, r]):
                    if (features[c, r, k] != features[c, r - 1, k] or
                        states[c, r, k] != states[c, r - 1, k] or
                        modifiers[c, r, k] != modifiers[c, r - 1, k]):
                        same = False
                        break
                if same:
                    valid[c] = False
                    break
    return valid


class RandomFuzzyRulesClassifier(ClassifierMixin, BaseEstimator):
    """Interpretable fuzzy-rule classifier using independent random RuleSets.

    Candidate quotas are approximately uniform over n_rules=1..max_rules.
    Rule lengths are sampled uniformly from 1..max_rules_len. Sampling stops
    after obtaining n_candidates unique canonical RuleSets or after
    max_sampling_attempts raw attempts. All unique candidates are evaluated in
    one parallel Numba call. Accuracy ties are resolved by model simplicity.
    """

    def __init__(
        self,
        max_rules: int = 6,
        max_rules_len: int = 6,
        max_rule_len: Optional[int] = None,
        max_literal_repetitions: int = 3,
        threshold: float = 0.5,
        n_candidates: int = 20_000,
        max_sampling_attempts: int = 1_000_000,
        sampling_chunk_size: int = 20_000,
        class_names: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
        categorical_features: Optional[Union[Sequence[int], Sequence[bool], Sequence[str]]] = None,
        continuous_features: Optional[Union[str, Sequence[int], Sequence[bool], Sequence[str]]] = None,
        categorical_feature_groups: Optional[Sequence[Sequence[int]]] = None,
        preprocessed: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.max_rules = max_rules
        self.max_rules_len = max_rules_len
        self.max_rule_len = max_rule_len
        self.max_literal_repetitions = max_literal_repetitions
        self.threshold = threshold
        self.n_candidates = n_candidates
        self.max_sampling_attempts = max_sampling_attempts
        self.sampling_chunk_size = sampling_chunk_size
        self.class_names = class_names
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.categorical_feature_groups = categorical_feature_groups
        self.preprocessed = preprocessed
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self._validate_parameters()
        self._check_X_y_no_return(X, y)
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        if len(self.classes_) != 2:
            raise ValueError("RandomFuzzyRulesClassifier supports binary classification only.")
        self.negative_class_, self.positive_class_ = self.classes_
        y_binary = np.ascontiguousarray(y_array == self.positive_class_, dtype=np.bool_)
        self.class_prior_positive_ = float(y_binary.mean())
        if self.class_names is not None and len(self.class_names) != 2:
            raise ValueError("class_names must have length 2")

        self._input_feature_names_ = self._resolve_input_feature_names(X)
        self.n_features_in_ = len(self._input_feature_names_)

        if self.preprocessed:
            Xt = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
            if Xt.ndim != 2 or not np.isfinite(Xt).all():
                raise ValueError("Preprocessed X must be a finite two-dimensional array.")
            if np.any(Xt < 0.0) or np.any(Xt > 1.0):
                raise ValueError("Preprocessed X values must lie in [0, 1].")
            self.preprocessor_ = None
            self.n_transformed_features_ = Xt.shape[1]
            names = list(self.feature_names) if self.feature_names is not None else [f"x{i+1}" for i in range(Xt.shape[1])]
            if len(names) != Xt.shape[1]:
                raise ValueError("feature_names must match transformed columns")
            self.transformed_feature_names_ = names
            self._configure_preprocessed_metadata(Xt.shape[1], names)
        else:
            cat_cols, num_cols = self._resolve_categorical_and_numeric_columns(X)
            self.categorical_columns_, self.numeric_columns_ = cat_cols, num_cols
            transformers = []
            if num_cols:
                transformers.append(("num", MinMaxScaler(clip=True), num_cols))
            if cat_cols:
                transformers.append(("cat", self._make_one_hot_encoder(), cat_cols))
            if not transformers:
                raise ValueError("No input columns available for preprocessing")
            self.preprocessor_ = ColumnTransformer(transformers, remainder="drop")
            Xt = np.ascontiguousarray(self.preprocessor_.fit_transform(X), dtype=np.float64)
            self.n_transformed_features_ = Xt.shape[1]
            self._build_internal_metadata()

        start = time.perf_counter()
        encoded_candidates = self._sample_unique_candidates_encoded()
        self.sampling_time_ = time.perf_counter() - start
        features, states, modifiers, lengths, n_rules = encoded_candidates
        if features.shape[0] == 0:
            raise RuntimeError("Random search generated no valid candidates.")

        start = time.perf_counter()
        if self.threshold == 0.5:
            accuracies = _batch_accuracy_threshold_half(Xt, y_binary, features, states, modifiers, lengths, n_rules)
        else:
            accuracies = _batch_accuracy(Xt, y_binary, features, states, modifiers, lengths, n_rules, self.threshold)
        self.evaluation_time_ = time.perf_counter() - start
        best_index = self._select_best_encoded(
            accuracies, modifiers, lengths, n_rules
        )
        self.rules_struct_ = self._decode_ruleset(
            features[best_index], states[best_index], modifiers[best_index],
            lengths[best_index], int(n_rules[best_index])
        )
        self.train_acc_ = float(accuracies[best_index])
        self.rule_stats_ = self._compute_rule_stats(Xt, y_binary, self.rules_struct_)
        self.rules_ = self._format_rules(self.rules_struct_, self.rule_stats_)
        self._compiled_rule_arrays_ = self._encode_single_ruleset(self.rules_struct_)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["rules_struct_", "_compiled_rule_arrays_"])
        self._check_X_no_return(X)
        Xt = np.ascontiguousarray(np.asarray(X, dtype=np.float64) if self.preprocessed else self.preprocessor_.transform(X), dtype=np.float64)
        return _score_rules_fast(Xt, *self._compiled_rule_arrays_)

    def predict_proba(self, X):
        score = self.decision_function(X)
        p = 1.0 - np.exp(2.0 * np.log(0.5) * score)
        return np.column_stack((1.0 - p, p))

    def predict(self, X):
        positive = self.predict_proba(X)[:, 1] >= self.threshold
        return np.where(positive, self.positive_class_, self.negative_class_)

    def _sample_unique_candidates_encoded(self):
        """Generate and deduplicate canonical candidates without Python Conditions."""
        rng = np.random.default_rng(self.random_state)
        quotas = np.full(self.max_rules, self.n_candidates // self.max_rules, dtype=int)
        quotas[: self.n_candidates % self.max_rules] += 1
        attempt_quotas = np.full(self.max_rules, self.max_sampling_attempts // self.max_rules, dtype=int)
        attempt_quotas[: self.max_sampling_attempts % self.max_rules] += 1

        continuous_mask = np.zeros(self.n_transformed_features_, dtype=np.bool_)
        continuous_mask[list(self.continuous_feature_indices_)] = True
        group_ids = np.full(self.n_transformed_features_, -1, dtype=np.int64)
        for group_id, group in enumerate(self.categorical_feature_groups_):
            for feature in group:
                group_ids[feature] = group_id

        kept_f, kept_s, kept_m, kept_l, kept_nr = [], [], [], [], []
        total_attempts = 0
        total_valid = 0

        for n_rules_value in range(1, self.max_rules + 1):
            target = int(quotas[n_rules_value - 1])
            attempt_limit = int(attempt_quotas[n_rules_value - 1])
            attempts = 0
            keys = set()
            sf, sz, sm, sl = [], [], [], []

            while len(keys) < target and attempts < attempt_limit:
                n_raw = min(self.sampling_chunk_size, attempt_limit - attempts)
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                f, st, mod, lengths = _generate_raw_candidates_fast(
                    seed, n_raw, n_rules_value, self._effective_max_rules_len_,
                    self.max_literal_repetitions, continuous_mask
                )
                valid = _canonicalize_raw_candidates_fast(
                    f, st, mod, lengths, n_rules_value, group_ids
                )
                attempts += n_raw
                total_attempts += n_raw

                for c in np.flatnonzero(valid):
                    total_valid += 1
                    key = (
                        f[c].tobytes() + st[c].tobytes() + mod[c].tobytes()
                        + lengths[c].tobytes()
                    )
                    if key in keys:
                        continue
                    keys.add(key)
                    sf.append(f[c].copy())
                    sz.append(st[c].copy())
                    sm.append(mod[c].copy())
                    sl.append(lengths[c].copy())
                    if len(keys) >= target:
                        break

            if sf:
                # Pad every stratum to max_rules so all strata can be
                # concatenated and evaluated in one Numba call.
                n_kept = len(sf)
                padded_f = np.full(
                    (n_kept, self.max_rules, self._effective_max_rules_len_),
                    -1, dtype=np.int64
                )
                padded_s = np.zeros_like(padded_f, dtype=np.uint8)
                padded_m = np.ones_like(padded_f, dtype=np.uint8)
                padded_l = np.zeros((n_kept, self.max_rules), dtype=np.int64)
                padded_f[:, :n_rules_value] = np.stack(sf)
                padded_s[:, :n_rules_value] = np.stack(sz)
                padded_m[:, :n_rules_value] = np.stack(sm)
                padded_l[:, :n_rules_value] = np.stack(sl)
                kept_f.append(padded_f)
                kept_s.append(padded_s)
                kept_m.append(padded_m)
                kept_l.append(padded_l)
                kept_nr.append(np.full(n_kept, n_rules_value, dtype=np.int64))
            if self.verbose:
                print(
                    f"[Random n_rules={n_rules_value}] unique={len(keys)}/{target}, "
                    f"attempts={attempts}/{attempt_limit}"
                )

        if kept_f:
            features = np.concatenate(kept_f)
            states = np.concatenate(kept_s)
            modifiers = np.concatenate(kept_m)
            lengths = np.concatenate(kept_l)
            n_rules = np.concatenate(kept_nr)
        else:
            shape = (0, self.max_rules, self._effective_max_rules_len_)
            features = np.empty(shape, dtype=np.int64)
            states = np.empty(shape, dtype=np.uint8)
            modifiers = np.empty(shape, dtype=np.uint8)
            lengths = np.empty((0, self.max_rules), dtype=np.int64)
            n_rules = np.empty(0, dtype=np.int64)

        self.n_sampling_attempts_ = total_attempts
        self.n_unique_candidates_ = features.shape[0]
        self.n_valid_before_dedup_ = total_valid
        self.duplicate_rate_ = (
            1.0 - self.n_unique_candidates_ / total_valid if total_valid else 0.0
        )
        if self.n_unique_candidates_ < self.n_candidates:
            warnings.warn(
                f"Generated {self.n_unique_candidates_} unique candidates instead "
                f"of {self.n_candidates} after {total_attempts} attempts.",
                UserWarning,
            )
        return features, states, modifiers, lengths, n_rules

    @staticmethod
    def _select_best_encoded(accuracies, modifiers, lengths, n_rules):
        best = 0
        for i in range(1, len(accuracies)):
            better_accuracy = accuracies[i] > accuracies[best] + 1e-15
            same_accuracy = abs(accuracies[i] - accuracies[best]) <= 1e-15
            if better_accuracy:
                best = i
            elif same_accuracy:
                complexity_i = (
                    int(lengths[i].sum()), int(n_rules[i]),
                    int(modifiers[i, :n_rules[i]].sum())
                )
                complexity_best = (
                    int(lengths[best].sum()), int(n_rules[best]),
                    int(modifiers[best, :n_rules[best]].sum())
                )
                if complexity_i < complexity_best:
                    best = i
        return best

    def _decode_ruleset(self, features, states, modifiers, lengths, n_rules):
        rules = []
        for r in range(n_rules):
            rule = tuple(
                Condition(
                    int(features[r, k]), self._code_to_state(int(states[r, k])),
                    int(modifiers[r, k])
                )
                for k in range(int(lengths[r]))
            )
            rules.append(rule)
        return tuple(rules)


    @staticmethod
    def _code_to_state(code):
        return ("high", "low", "medium", "present", "absent")[code]

    @staticmethod
    def _canonical_ruleset_key(rules):
        return (len(rules), tuple((len(rule), rule) for rule in rules))

    def _encode_single_ruleset(self, rules):
        max_len = max((len(rule) for rule in rules), default=0)
        f = np.full((len(rules), max_len), -1, dtype=np.int64)
        s = np.zeros((len(rules), max_len), dtype=np.uint8)
        m = np.ones((len(rules), max_len), dtype=np.uint8)
        lengths = np.zeros(len(rules), dtype=np.int64)
        for r, rule in enumerate(rules):
            lengths[r] = len(rule)
            for k, condition in enumerate(rule):
                f[r, k], s[r, k], m[r, k] = condition.feature, _STATE_TO_CODE[condition.state], condition.modifier
        return f, s, m, lengths

    def _compute_rule_stats(self, X, y, rules):
        stats = []
        prior = max(float(np.mean(y)), np.finfo(float).eps)
        for rule in rules:
            covered = np.ones(X.shape[0], dtype=bool)
            for condition in rule:
                x = X[:, condition.feature]
                if condition.state == "high":
                    covered &= x >= 0.5 ** (1.0 / condition.modifier)
                elif condition.state == "low":
                    covered &= x <= 1.0 - 0.5 ** (1.0 / condition.modifier)
                elif condition.state == "medium":
                    covered &= (x >= 0.25) & (x <= 0.75)
                elif condition.state == "present":
                    covered &= x >= 0.5
                else:
                    covered &= x < 0.5
            n_cov = int(covered.sum())
            n_pos = int(np.sum(y[covered])) if n_cov else 0
            precision = n_pos / n_cov if n_cov else 0.0
            stats.append(RuleStats(n_cov / len(y), precision / prior if n_cov else 0.0, n_cov, n_pos, n_cov - n_pos))
        return stats

    def _format_rules(self, rules, stats):
        positive = self.class_names[1] if self.class_names is not None else "1"
        negative = self.class_names[0] if self.class_names is not None else "0"
        output = []
        for rule, stat in zip(rules, stats):
            antecedent = " AND ".join(self._condition_to_text(c) for c in rule)
            output.append(
                f"IF {antecedent} THEN class is {positive} | Lift: {stat.lift:.4f}; "
                f"Support: {stat.support:.4f}; Covered: {stat.n_covered}"
            )
        output.append(f"ELSE class is {negative}")
        return output

    def _condition_to_text(self, condition):
        meta = self.transformed_feature_metadata_[condition.feature]
        name = meta["name"]
        if condition.state in {"high", "low"}:
            prefix = {1: "", 2: "Very ", 3: "Extremely "}[condition.modifier]
            return f"{name} is {prefix}{condition.state.title()}"
        if condition.state == "medium":
            return f"{name} is Medium"
        category = meta["category"]
        return f"{name} is {category}" if condition.state == "present" else f"{name} is not {category}"

    def _validate_parameters(self):
        self._effective_max_rules_len_ = self.max_rule_len if self.max_rule_len is not None else self.max_rules_len
        if min(self.max_rules, self._effective_max_rules_len_, self.n_candidates, self.max_sampling_attempts, self.sampling_chunk_size) < 1:
            raise ValueError("Size and budget parameters must be >= 1")
        if not 1 <= self.max_literal_repetitions <= 3:
            raise ValueError("max_literal_repetitions must be in [1, 3]")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.preprocessed and self.continuous_features is None:
            raise ValueError("continuous_features is required when preprocessed=True; use 'all' when appropriate")

    def _configure_preprocessed_metadata(self, n_features, names):
        if self.continuous_features == "all":
            continuous, groups = set(range(n_features)), []
        else:
            continuous = self._resolve_index_spec(self.continuous_features, n_features, names, "continuous_features")
            if self.categorical_feature_groups is None:
                if continuous != set(range(n_features)):
                    raise ValueError("categorical_feature_groups is required for non-continuous transformed columns")
                groups = []
            else:
                groups = [frozenset(map(int, group)) for group in self.categorical_feature_groups]
        flat = [j for group in groups for j in group]
        if len(flat) != len(set(flat)) or any(j < 0 or j >= n_features for j in flat):
            raise ValueError("categorical_feature_groups must be valid and non-overlapping")
        categorical = set(flat)
        if continuous & categorical:
            raise ValueError("Continuous and categorical features overlap")
        uncovered = set(range(n_features)) - continuous - categorical
        if uncovered:
            raise ValueError(f"Unspecified transformed features: {sorted(uncovered)}")
        self.continuous_feature_indices_ = frozenset(continuous)
        self.categorical_feature_groups_ = tuple(groups)
        self.transformed_feature_metadata_ = [
            {"type": "numeric", "name": name, "category": None}
            if j in continuous else {"type": "categorical", "name": name, "category": "present"}
            for j, name in enumerate(names)
        ]
        self.numeric_columns_ = sorted(continuous)
        self.categorical_columns_ = sorted(categorical)

    def _build_internal_metadata(self):
        metadata, continuous, groups = [], [], []
        j = 0
        if "num" in self.preprocessor_.named_transformers_:
            for column in self.numeric_columns_:
                original = self._column_to_index(column)
                metadata.append({"type": "numeric", "name": self._input_feature_names_[original], "category": None})
                continuous.append(j); j += 1
        if "cat" in self.preprocessor_.named_transformers_:
            encoder = self.preprocessor_.named_transformers_["cat"]
            for column, categories in zip(self.categorical_columns_, encoder.categories_):
                original = self._column_to_index(column)
                group = []
                for category in categories:
                    metadata.append({"type": "categorical", "name": self._input_feature_names_[original], "category": str(category)})
                    group.append(j); j += 1
                groups.append(frozenset(group))
        self.transformed_feature_metadata_ = metadata
        self.transformed_feature_names_ = [m["name"] if m["type"] == "numeric" else f'{m["name"]}={m["category"]}' for m in metadata]
        self.continuous_feature_indices_ = frozenset(continuous)
        self.categorical_feature_groups_ = tuple(groups)

    @staticmethod
    def _resolve_index_spec(spec, n_features, names, parameter_name):
        values = list(spec)
        if len(values) == n_features and all(isinstance(v, (bool, np.bool_)) for v in values):
            return {i for i, v in enumerate(values) if v}
        if all(isinstance(v, (int, np.integer)) for v in values):
            result = {int(v) for v in values}
        elif all(isinstance(v, str) for v in values):
            mapping = {name: i for i, name in enumerate(names)}
            unknown = [v for v in values if v not in mapping]
            if unknown:
                raise ValueError(f"Unknown names in {parameter_name}: {unknown}")
            result = {mapping[v] for v in values}
        else:
            raise ValueError(f"{parameter_name} must be indices, names, or a boolean mask")
        if any(i < 0 or i >= n_features for i in result):
            raise ValueError(f"{parameter_name} contains invalid indices")
        return result

    def _resolve_input_feature_names(self, X):
        if self.feature_names is not None:
            names = list(self.feature_names)
        elif pd is not None and isinstance(X, pd.DataFrame):
            names = list(map(str, X.columns))
        else:
            names = [f"x{i+1}" for i in range(X.shape[1])]
        if len(names) != X.shape[1]:
            raise ValueError("feature_names length mismatch")
        return names

    def _resolve_categorical_and_numeric_columns(self, X):
        n, names = X.shape[1], self._input_feature_names_
        if self.categorical_features is None:
            categorical = [i for i, column in enumerate(X.columns) if str(X[column].dtype) in ("object", "category", "bool")] if pd is not None and isinstance(X, pd.DataFrame) else []
        else:
            categorical = sorted(self._resolve_index_spec(self.categorical_features, n, names, "categorical_features"))
        numerical = [i for i in range(n) if i not in set(categorical)]
        if pd is not None and isinstance(X, pd.DataFrame) and self.feature_names is None:
            return [X.columns[i] for i in categorical], [X.columns[i] for i in numerical]
        return categorical, numerical

    def _make_one_hot_encoder(self):
        try:
            return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover
            return OneHotEncoder(sparse=False, handle_unknown="ignore")

    def _column_to_index(self, column):
        return int(column) if isinstance(column, (int, np.integer)) else self._input_feature_names_.index(str(column))

    def _check_X_y_no_return(self, X, y):
        try:
            check_X_y(X, y, dtype=None, ensure_all_finite=True)
        except TypeError:  # older sklearn
            check_X_y(X, y, dtype=None, force_all_finite=True)

    def _check_X_no_return(self, X):
        try:
            check_array(X, dtype=None, ensure_all_finite=True)
        except TypeError:
            check_array(X, dtype=None, force_all_finite=True)
