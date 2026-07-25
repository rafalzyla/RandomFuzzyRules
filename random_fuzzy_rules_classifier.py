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

@njit(cache=True, fastmath=False, inline="always", forceinline=True)
def _sample_structure_size(max_value, sampling_type):
    """Sample a rule count or rule length.

    Parameters
    ----------
    max_value : int
        Maximum allowed value. The returned value belongs to
        {1, ..., max_value}.

    sampling_type : int
        Sampling strategy:

        1 : discrete uniform
            Uniformly sample from {1, ..., max_value}.

        2 : exponential/log-uniform scale
            floor(2 ** U(0, log2(max_value + 1))).

    Returns
    -------
    value : int
        Sampled integer in {1, ..., max_value}.
    """
    if max_value <= 1:
        return 1

    if sampling_type == 1:
        return np.random.randint(1, max_value + 1)

    # sampling_type == 2
    value = int(2.0 ** np.random.uniform(0.0, np.log2(max_value + 1.0)))

    # Defensive protection against floating-point rounding.
    if value > max_value:
        value = max_value

    return value

@njit(cache=True, fastmath=False, inline="always", forceinline=True)
def _hash_candidate_fast(features, states, modifiers, lengths, n_rules,):
    """Calculate a deterministic 64-bit hash of a canonical RuleSet."""
    hash_value = np.uint64(1469598103934665603)

    hash_prime = np.uint64(1099511628211)

    hash_value = (hash_value ^ np.uint64(n_rules)) * hash_prime

    for rule_index in range(n_rules):
        rule_length = lengths[rule_index]

        hash_value = (hash_value ^ np.uint64(rule_length)) * hash_prime

        for literal_index in range(rule_length):
            # Add one so that the value zero does not behave
            # like an empty/padding value in the hash stream.
            hash_value = (hash_value ^ np.uint64(features[rule_index, literal_index] + 1)) * hash_prime
            hash_value = (hash_value ^ np.uint64(states[rule_index, literal_index] + 1)) * hash_prime
            hash_value = (hash_value ^ np.uint64(modifiers[rule_index, literal_index] + 1)) * hash_prime

    return hash_value

@njit(cache=True, fastmath=False, inline="always", forceinline=True)
def _candidate_equals_stored_fast(
    stored_features,
    stored_states,
    stored_modifiers,
    stored_lengths,
    stored_n_rules,
    stored_index,
    candidate_features,
    candidate_states,
    candidate_modifiers,
    candidate_lengths,
    candidate_n_rules,
):
    """Compare a temporary candidate with an accepted candidate."""
    if stored_n_rules[stored_index] != candidate_n_rules:
        return False

    for rule_index in range(candidate_n_rules):
        rule_length = candidate_lengths[rule_index]

        if stored_lengths[stored_index, rule_index] != rule_length:
            return False

        for literal_index in range(rule_length):
            if stored_features[stored_index, rule_index, literal_index] != candidate_features[rule_index, literal_index]:
                return False

            if stored_states[stored_index, rule_index, literal_index] != candidate_states[rule_index, literal_index]:
                return False

            if stored_modifiers[stored_index, rule_index, literal_index] != candidate_modifiers[rule_index, literal_index]:
                return False

    return True

@njit(cache=True, fastmath=False, inline="always", forceinline=True)
def _canonicalize_candidate_fast(features, states, modifiers, lengths, n_rules, group_ids):
    """Canonicalize and validate one temporary RuleSet in-place.

    Invalid candidates are rejected rather than shortened. This preserves
    exactly the sampled number of rules and sampled rule lengths.

    Returns
    -------
    valid : bool
        True if the candidate is structurally valid.
    """
    # --------------------------------------------------------------
    # Validate and sort conditions inside every rule.
    # --------------------------------------------------------------
    for rule_index in range(n_rules):
        rule_length = lengths[rule_index]

        for current_position in range(rule_length):
            current_feature = features[rule_index, current_position]
            current_state = states[rule_index, current_position]
            current_group = group_ids[current_feature]

            for previous_position in range(current_position):
                previous_feature = features[rule_index,previous_position]

                previous_state = states[rule_index, previous_position]

                # Only one semantic condition is allowed for a
                # transformed feature.
                if current_feature == previous_feature:
                    return False

                previous_group = group_ids[previous_feature]

                # Two conditions from the same one-hot group are
                # invalid/redundant if either requires "present".
                #
                # Multiple "absent" conditions from the same group
                # remain allowed.
                if (
                    current_group >= 0
                    and current_group == previous_group
                    and (current_state == 3 or previous_state == 3)
                ):
                    return False

        # Sort conditions by:
        # feature, state, modifier.
        for current_position in range(1, rule_length):
            current_feature = features[rule_index, current_position]
            current_state = states[rule_index, current_position]
            current_modifier = modifiers[rule_index, current_position]
            previous_position = current_position - 1

            while previous_position >= 0:
                previous_feature = features[rule_index, previous_position]
                previous_state = states[rule_index, previous_position]
                previous_modifier = modifiers[rule_index, previous_position]

                greater = (
                    previous_feature > current_feature
                    or (
                        previous_feature == current_feature
                        and (
                            previous_state > current_state
                            or (
                                previous_state == current_state
                                and previous_modifier > current_modifier
                            )
                        )
                    )
                )

                if not greater:
                    break

                features[rule_index, previous_position + 1] = previous_feature
                states[rule_index, previous_position + 1] = previous_state
                modifiers[rule_index, previous_position + 1] = previous_modifier
                previous_position -= 1

            features[rule_index, previous_position + 1] = current_feature
            states[rule_index, previous_position + 1] = current_state
            modifiers[rule_index, previous_position + 1] = current_modifier

    # --------------------------------------------------------------
    # Sort rules by:
    # rule length, then lexicographically by conditions.
    # --------------------------------------------------------------
    for current_rule in range(1, n_rules):
        current_length = lengths[current_rule]
        current_features = features[current_rule].copy()
        current_states = states[current_rule].copy()
        current_modifiers = modifiers[current_rule].copy()
        previous_rule = current_rule - 1

        while previous_rule >= 0:
            previous_length = lengths[previous_rule]

            greater = previous_length > current_length

            if previous_length == current_length:
                greater = False

                for literal_index in range(current_length):
                    if features[previous_rule, literal_index,] != current_features[literal_index]:
                        greater = features[previous_rule, literal_index] > current_features[literal_index]
                        break

                    if states[previous_rule, literal_index] != current_states[literal_index]:
                        greater = states[previous_rule, literal_index] > current_states[literal_index]
                        break

                    if modifiers[previous_rule, literal_index] != current_modifiers[literal_index]:
                        greater = modifiers[previous_rule,literal_index] > current_modifiers[literal_index]
                        break

            if not greater:
                break

            lengths[previous_rule + 1] = lengths[previous_rule]
            features[previous_rule + 1] = features[previous_rule]
            states[previous_rule + 1] = states[previous_rule]
            modifiers[previous_rule + 1] = modifiers[previous_rule]
            previous_rule -= 1

        lengths[previous_rule + 1] = current_length
        features[previous_rule + 1] = current_features
        states[previous_rule + 1] = current_states
        modifiers[previous_rule + 1] = current_modifiers

    # --------------------------------------------------------------
    # Reject duplicate rules.
    # --------------------------------------------------------------
    for rule_index in range(1, n_rules):
        if lengths[rule_index] != lengths[rule_index - 1]:
            continue

        same = True

        for literal_index in range(lengths[rule_index]):
            if (
                features[rule_index, literal_index] != features[rule_index - 1, literal_index,]
                or states[rule_index, literal_index] != states[rule_index - 1, literal_index]
                or modifiers[rule_index, literal_index] != modifiers[rule_index - 1, literal_index]
            ):
                same = False
                break

        if same:
            return False

    return True

@njit(cache=True, fastmath=False, parallel=False,)
def _generate_unique_candidates_fast(
    seed,
    n_candidates,
    max_sampling_attempts,
    max_rules,
    max_rule_length,
    max_modifier,
    continuous_mask,
    group_ids,
    sampling_type_number,
    sampling_type_length,
):
    """Generate unique canonical RuleSets in a single Numba call.

    Parameters
    ----------
    seed : int
        Random seed.

    n_candidates : int
        Target number of unique valid RuleSets.

    max_sampling_attempts : int
        Maximum number of raw candidate draws.

    max_rules : int
        Maximum number of rules in a RuleSet.

    max_rule_length : int
        Maximum number of conditions in a rule.

    max_modifier : int
        Maximum High/Low modifier.

    continuous_mask : ndarray of bool
        Mask identifying continuous transformed features.

    group_ids : ndarray of int64
        One-hot group identifier for every transformed feature.
        Continuous features use -1.

    sampling_type_number : {1, 2}
        Sampling strategy for the number of rules:

        1 : discrete uniform
        2 : exponential/log-uniform scale

    sampling_type_length : {1, 2}
        Sampling strategy for rule lengths:

        1 : discrete uniform
        2 : exponential/log-uniform scale

    Returns
    -------
    features, states, modifiers, lengths, n_rules
        Encoded accepted candidates.

    attempts : int
        Number of raw sampling attempts.

    invalid_count : int
        Number of structurally invalid candidates.

    duplicate_count : int
        Number of duplicate candidates.
    """
    np.random.seed(seed)

    n_features = continuous_mask.shape[0]

    # Final output arrays are allocated only once.
    output_features = np.full((n_candidates, max_rules, max_rule_length), -1, dtype=np.int64)
    output_states = np.zeros((n_candidates, max_rules, max_rule_length), dtype=np.uint8)
    output_modifiers = np.ones((n_candidates, max_rules, max_rule_length), dtype=np.uint8)
    output_lengths = np.zeros((n_candidates, max_rules), dtype=np.int64)
    output_n_rules = np.zeros(n_candidates, dtype=np.int64)

    # Hash-table size is the next power of two >= 2 * n_candidates.
    # This keeps the load factor at or below 0.5.
    hash_table_size = 1

    while (hash_table_size < 2 * n_candidates):
        hash_table_size *= 2

    stored_hashes = np.zeros(hash_table_size, dtype=np.uint64)

    stored_indices = np.full(hash_table_size, -1, dtype=np.int64)

    # Reusable temporary candidate buffers.
    candidate_features = np.full((max_rules, max_rule_length), -1, dtype=np.int64)
    candidate_states = np.zeros((max_rules, max_rule_length), dtype=np.uint8)
    candidate_modifiers = np.ones((max_rules,max_rule_length), dtype=np.uint8)
    candidate_lengths = np.zeros(max_rules, dtype=np.int64)

    accepted_count = 0
    attempts = 0
    invalid_count = 0
    duplicate_count = 0

    while (accepted_count < n_candidates and attempts < max_sampling_attempts):
        attempts += 1

        # Reset the reusable temporary buffers.
        candidate_features[:, :] = -1
        candidate_states[:, :] = 0
        candidate_modifiers[:, :] = 1
        candidate_lengths[:] = 0

        candidate_n_rules =  _sample_structure_size(max_rules, sampling_type_number)

        # ----------------------------------------------------------
        # Generate one complete raw RuleSet.
        # ----------------------------------------------------------
        for rule_index in range(candidate_n_rules):
            rule_length = _sample_structure_size(max_rule_length, sampling_type_length)

            candidate_lengths[rule_index] = rule_length

            for literal_index in range(rule_length):
                feature = np.random.randint(0, n_features)
                candidate_features[rule_index, literal_index] = feature

                if continuous_mask[feature]:
                    state = np.random.randint(0, 3)
                    candidate_states[rule_index,literal_index] = state

                    if state < 2:
                        candidate_modifiers[rule_index, literal_index] = np.random.randint(1, max_modifier + 1)

                else:
                    candidate_states[rule_index, literal_index] = np.random.randint(3, 5)

        # ----------------------------------------------------------
        # Canonicalize and validate immediately.
        # ----------------------------------------------------------
        valid = _canonicalize_candidate_fast(
            candidate_features,
            candidate_states,
            candidate_modifiers,
            candidate_lengths,
            candidate_n_rules,
            group_ids,
        )

        if not valid:
            invalid_count += 1
            continue

        # ----------------------------------------------------------
        # Exact deduplication with a hash table.
        # ----------------------------------------------------------
        candidate_hash = (
            _hash_candidate_fast(
                candidate_features,
                candidate_states,
                candidate_modifiers,
                candidate_lengths,
                candidate_n_rules,
            )
        )

        table_position = np.int64(candidate_hash & np.uint64(hash_table_size - 1))

        duplicate = False

        while (stored_indices[table_position] != -1):
            stored_index = stored_indices[table_position]

            if (
                stored_hashes[table_position] == candidate_hash
                and _candidate_equals_stored_fast(
                    output_features,
                    output_states,
                    output_modifiers,
                    output_lengths,
                    output_n_rules,
                    stored_index,
                    candidate_features,
                    candidate_states,
                    candidate_modifiers,
                    candidate_lengths,
                    candidate_n_rules,
                )
            ):
                duplicate = True
                break

            table_position = np.int64((table_position + 1) & (hash_table_size - 1))

        if duplicate:
            duplicate_count += 1
            continue

        # ----------------------------------------------------------
        # Store the accepted candidate.
        # ----------------------------------------------------------
        output_features[accepted_count] = candidate_features
        output_states[accepted_count] = candidate_states
        output_modifiers[accepted_count] = candidate_modifiers
        output_lengths[accepted_count] = candidate_lengths
        output_n_rules[accepted_count] = candidate_n_rules
        stored_hashes[table_position] = candidate_hash
        stored_indices[table_position] = accepted_count
        accepted_count += 1

    return (
        output_features[:accepted_count],
        output_states[:accepted_count],
        output_modifiers[:accepted_count],
        output_lengths[:accepted_count],
        output_n_rules[:accepted_count],
        attempts,
        invalid_count,
        duplicate_count,
    )

class RandomFuzzyRulesClassifier(ClassifierMixin, BaseEstimator):
    """Interpretable fuzzy-rule classifier using random RuleSets.
    
    Candidates are generated one at a time in a compiled Numba function.
    Each candidate is immediately canonicalized, validated and checked for
    duplication. Sampling stops after obtaining n_candidates unique valid
    RuleSets or after max_sampling_attempts raw draws.
    
    The number of rules and rule lengths can be sampled independently:
    
        sampling_type_number = 1
            Discrete uniform number of rules.
    
        sampling_type_number = 2
            Exponential/log-uniform number of rules.
    
        sampling_type_length = 1
            Discrete uniform rule lengths.
    
        sampling_type_length = 2
            Exponential/log-uniform rule lengths.
    
    All accepted candidates are evaluated in one parallel Numba call.
    Accuracy ties are resolved by model simplicity.
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
        sampling_type_number: int = 1,
        sampling_type_length: int = 1,
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
        self.sampling_type_number = sampling_type_number
        self.sampling_type_length = sampling_type_length
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
        """Generate unique candidates in one compiled Numba call."""
        continuous_mask = np.zeros(self.n_transformed_features_, dtype=np.bool_)
        continuous_mask[list(self.continuous_feature_indices_)] = True
    
        group_ids = np.full(self.n_transformed_features_, -1, dtype=np.int64)
    
        for group_id, group in enumerate(self.categorical_feature_groups_):
            for feature in group:
                group_ids[feature] = group_id
    
        # np.random.seed used by the Numba generator requires an integer.
        # Preserve nondeterministic sklearn semantics for random_state=None.
        if self.random_state is None:
            seed = int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        else:
            seed = int(self.random_state)
    
        (
            features,
            states,
            modifiers,
            lengths,
            n_rules,
            attempts,
            invalid_count,
            duplicate_count,
        ) = _generate_unique_candidates_fast(
            seed=seed,
            n_candidates=self.n_candidates,
            max_sampling_attempts=self.max_sampling_attempts,
            max_rules=self.max_rules,
            max_rule_length=self._effective_max_rules_len_,
            max_modifier=self.max_literal_repetitions,
            continuous_mask=continuous_mask,
            group_ids=group_ids,
            sampling_type_number=self.sampling_type_number,
            sampling_type_length=self.sampling_type_length,
        )
    
        self.n_sampling_attempts_ = int(attempts)
        self.n_unique_candidates_ = int(features.shape[0])
        self.n_invalid_candidates_ = int(invalid_count)
        self.n_duplicate_candidates_ = int(duplicate_count)
        self.valid_sampling_attempts_ = self.n_sampling_attempts_ - self.n_invalid_candidates_
        self.duplicate_rate_ = self.n_duplicate_candidates_ / self.valid_sampling_attempts_ if self.valid_sampling_attempts_ > 0 else 0.0
        self.invalid_rate_ = self.n_invalid_candidates_ / self.n_sampling_attempts_ if self.n_sampling_attempts_ > 0 else 0.0
    
        if self.n_unique_candidates_ < self.n_candidates:
            warnings.warn(
                f"Generated "
                f"{self.n_unique_candidates_} "
                f"unique candidates instead of "
                f"{self.n_candidates} after "
                f"{self.n_sampling_attempts_} "
                "attempts.",
                UserWarning,
            )
    
        return features, states, modifiers, lengths, n_rules

    @staticmethod
    def _select_best_encoded(accuracies, modifiers, lengths, n_rules):
        best = 0
    
        best_modifier_sum = 0
    
        for rule_index in range(n_rules[0]):
            for literal_index in range(lengths[0, rule_index]):
                best_modifier_sum += int(modifiers[0, rule_index, literal_index])
    
        for candidate_index in range(1, len(accuracies)):
            better_accuracy = accuracies[candidate_index] > accuracies[best] + 1e-15
    
            same_accuracy = abs(accuracies[candidate_index] - accuracies[best]) <= 1e-15
    
            if better_accuracy:
                best = candidate_index
    
                best_modifier_sum = 0
    
                for rule_index in range(n_rules[best]):
                    for literal_index in range(lengths[best, rule_index]):
                        best_modifier_sum += int(modifiers[best, rule_index, literal_index])
    
            elif same_accuracy:
                candidate_modifier_sum = 0
    
                for rule_index in range(n_rules[candidate_index]):
                    for literal_index in range(lengths[candidate_index, rule_index]):
                        candidate_modifier_sum += int(modifiers[candidate_index, rule_index, literal_index])
    
                complexity_candidate = (
                    int(lengths[candidate_index].sum()),
                    int(n_rules[candidate_index]),
                    candidate_modifier_sum,
                )
    
                complexity_best = (
                    int(lengths[best].sum()),
                    int(n_rules[best]),
                    best_modifier_sum,
                )
    
                if complexity_candidate < complexity_best:
                    best = candidate_index
                    best_modifier_sum = candidate_modifier_sum
    
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
    
        if min(self.max_rules, self._effective_max_rules_len_, self.n_candidates, self.max_sampling_attempts,) < 1:
            raise ValueError("Size and budget parameters must be >= 1.")
    
        if not (1 <= self.max_literal_repetitions <= 3):
            raise ValueError("max_literal_repetitions must be in [1, 3].")
    
        if self.sampling_type_number not in {1, 2}:
            raise ValueError("sampling_type_number must be 1 (uniform) or 2 (exponential).")
    
        if self.sampling_type_length not in {1, 2}:
            raise ValueError("sampling_type_length must be 1 (uniform) or 2 (exponential).")
    
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1].")
    
        if self.preprocessed and self.continuous_features is None:
            raise ValueError("continuous_features is required when preprocessed=True; use 'all' when appropriate.")

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
