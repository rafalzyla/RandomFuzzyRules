# ==================================================================
# The code was written in cooperation with GPT-{5.5, 5.6} Thinking
# ==================================================================

"""
BeamFuzzyRulesClassifier
========================

A prototype interpretable binary classifier inspired by fuzzy-rule GPR-style
models. The model searches for a compact disjunction of positive-class rules
using beam search. Each rule is a conjunction of literals. Numeric variables are
scaled to [0, 1]; categorical variables are one-hot encoded. A positive numeric
literal is interpreted as "High" and a negated numeric literal as "Low".

The classifier returns a fuzzy score

    S(x) = sum_k product_{literal in rule_k} activation(literal, x)

and a probability-like score

    p(class=positive | x) = 1 - exp(2 * ln(0.5) * S(x)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
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


# Numerical condition states: high, low, medium.
# Categorical condition states: present, absent.
_STATE_TO_CODE = {
    "high": 0,
    "low": 1,
    "medium": 2,
    "present": 3,
    "absent": 4,
}


@dataclass(frozen=True, order=True)
class Condition:
    """One interpretable antecedent condition.

    Parameters
    ----------
    feature : int
        Index of a transformed feature column.
    state : {'high', 'low', 'medium', 'present', 'absent'}
        Linguistic or categorical state.
    modifier : int, default=1
        For high/low: 1=plain, 2=very, 3=extremely. It must be 1 for
        medium/present/absent.
    """

    feature: int
    state: str
    modifier: int = 1

    def __post_init__(self):
        if self.state not in _STATE_TO_CODE:
            raise ValueError(f"Unknown condition state: {self.state!r}")
        if self.modifier < 1 or self.modifier > 3:
            raise ValueError("Condition modifier must be between 1 and 3.")
        if self.state in {"medium", "present", "absent"} and self.modifier != 1:
            raise ValueError(
                f"State {self.state!r} only supports modifier=1."
            )


Rule = Tuple[Condition, ...]
RuleSet = Tuple[Rule, ...]


@njit(cache=True, fastmath=False, parallel=False)
def _score_rules_fast(
    Xt,
    condition_features,
    condition_states,
    condition_modifiers,
    rule_lengths,
):
    """Compute S for one encoded rule set."""
    n_samples = Xt.shape[0]
    n_rules = rule_lengths.shape[0]
    out = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        total = 0.0
        for r in range(n_rules):
            activation = 1.0
            for k in range(rule_lengths[r]):
                j = condition_features[r, k]
                state = condition_states[r, k]
                modifier = condition_modifiers[r, k]
                x = Xt[i, j]

                if state == 0:  # high
                    if modifier == 1:
                        value = x
                    elif modifier == 2:
                        value = x * x
                    else:
                        value = x * x * x
                elif state == 1:  # low
                    low = 1.0 - x
                    if modifier == 1:
                        value = low
                    elif modifier == 2:
                        value = low * low
                    else:
                        value = low * low * low
                elif state == 2:  # medium
                    value = x * (1.0 - x)
                elif state == 3:  # categorical present
                    value = x
                else:  # categorical absent
                    value = 1.0 - x

                activation *= value
            total += activation
        out[i] = total
    return out


@njit(cache=True, fastmath=False, parallel=True)
def _batch_accuracy_fast(
    Xt,
    y,
    candidate_features,
    candidate_states,
    candidate_modifiers,
    candidate_rule_lengths,
    candidate_n_rules,
    threshold
):
    """Evaluate many rule sets in one parallel compiled call.
    """
    n_candidates = candidate_features.shape[0]
    n_samples = Xt.shape[0]
    accuracies = np.empty(n_candidates, dtype=np.float64)
    log_quarter = 2.0 * np.log(0.5)
    eps = np.finfo(np.float64).eps

    for c in prange(n_candidates):
        acc_sum = 0
        for i in range(n_samples):
            score = 0.0
            for r in range(candidate_n_rules[c]):
                activation = 1.0
                length = candidate_rule_lengths[c, r]
                for k in range(length):
                    j = candidate_features[c, r, k]
                    state = candidate_states[c, r, k]
                    modifier = candidate_modifiers[c, r, k]
                    x = Xt[i, j]

                    if state == 0:
                        if modifier == 1:
                            value = x
                        elif modifier == 2:
                            value = x * x
                        else:
                            value = x * x * x
                    elif state == 1:
                        low = 1.0 - x
                        if modifier == 1:
                            value = low
                        elif modifier == 2:
                            value = low * low
                        else:
                            value = low * low * low
                    elif state == 2:
                        value = x * (1.0 - x)
                    elif state == 3:
                        value = x
                    else:
                        value = 1.0 - x

                    activation *= value
                score += activation

            p = 1.0 - np.exp(log_quarter * score)
            pred = p >= threshold
            acc_sum += int(pred == y[i])
        accuracies[c] = acc_sum / n_samples
        
    return accuracies


@dataclass(frozen=True)
class RuleStats:
    support: float
    lift: float
    n_covered: int
    n_positive_covered: int
    n_negative_covered: int


class BeamFuzzyRulesClassifier(ClassifierMixin, BaseEstimator):
    """Interpretable fuzzy-rule binary classifier searched by beam search.

    Parameters
    ----------
    max_rules : int, default=5
        Maximum number of IF-THEN rules in the model.

    max_rules_len : int, default=3
        Maximum number of literals in a single rule antecedent.

    max_rule_len : int or None, default=None
        Backward-compatible alias for max_rules_len. If provided, it overrides
        max_rules_len.

    max_literal_repetitions : int, default=1
        Maximum modifier for numerical High/Low conditions. Must be 1, 2, or 3.

    threshold : float, default=0.5
        Decision threshold applied to the positive-class probability returned by
        ``predict_proba``.

    beam_width : int, default=3
        Number of best candidate models retained after every beam-search level.

    class_names : sequence of str, optional
        Human-readable class names. If provided, it should have length 2 and be
        ordered as [negative_class_name, positive_class_name]. If not provided,
        the displayed labels are "0" and "1".

    feature_names : sequence of str, optional
        Input feature names. If X is a pandas DataFrame and ``feature_names`` is
        None, the DataFrame column names are used. Otherwise names default to
        x1, x2, ..., xn.

    categorical_features : sequence, boolean mask, or None, default=None
        Categorical columns. May be specified as integer indices, boolean mask,
        or column names. If None and X is a DataFrame, object/category/bool
        columns are treated as categorical. If None and X is a NumPy array, all
        columns are treated as numeric.

    continuous_features : 'all', sequence of int/bool/str, or None
        Required when preprocessed=True. Use 'all' if every transformed feature
        is continuous. Otherwise supply continuous transformed-column indices.

    categorical_feature_groups : sequence of sequences, optional
        Groups of mutually exclusive one-hot transformed columns. Required when
        preprocessed=True and continuous_features does not cover every column.

    preprocessed : bool, default=False
        If True, X is assumed to be an already preprocessed,
        dense numerical matrix. Internal scaling and categorical
        encoding are skipped. This mode is intended for using the
        classifier inside an external sklearn Pipeline containing
        a ColumnTransformer.

    max_steps : int or None, default=None
        Maximum number of beam-expansion steps. If None, uses
        ``max_rules * max_rules_len``.

    random_state : int or None, default=None
        Present for sklearn-style compatibility. The current deterministic beam
        search does not use randomness.

    verbose : int, default=0
        Verbosity level. 0 is silent.
    """

    def __init__(
        self,
        max_rules: int = 5,
        max_rules_len: int = 3,
        max_rule_len: Optional[int] = None,
        max_literal_repetitions: int = 1,
        threshold: float = 0.5,
        beam_width: int = 3,
        class_names: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
        categorical_features: Optional[
            Union[Sequence[int], Sequence[bool], Sequence[str]]
        ] = None,
        continuous_features: Optional[
            Union[str, Sequence[int], Sequence[bool], Sequence[str]]
        ] = None,
        categorical_feature_groups: Optional[Sequence[Sequence[int]]] = None,
        preprocessed: bool = False,
        max_steps: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.max_rules = max_rules
        self.max_rules_len = max_rules_len
        self.max_rule_len = max_rule_len
        self.max_literal_repetitions = max_literal_repetitions
        self.threshold = threshold
        self.beam_width = beam_width
        self.class_names = class_names
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.categorical_feature_groups = categorical_feature_groups
        self.preprocessed = preprocessed
        self.max_steps = max_steps
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self._validate_parameters()
        self._check_X_y_no_return(X, y)

        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        if classes.shape[0] != 2:
            raise ValueError("BeamFuzzyRulesClassifier supports binary classification only.")
        self.classes_ = classes
        self.negative_class_ = classes[0]
        self.positive_class_ = classes[1]
        y_bin = (y_arr == self.positive_class_).astype(np.bool_)
        self.class_prior_positive_ = float(np.mean(y_bin))

        if self.class_names is not None and len(self.class_names) != 2:
            raise ValueError("class_names must have length 2: [negative, positive].")

        self._input_feature_names_ = self._resolve_input_feature_names(X)
        self.n_features_in_ = len(self._input_feature_names_)

        if self.preprocessed:
            Xt = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
            if Xt.ndim != 2:
                raise ValueError("When preprocessed=True, X must be two-dimensional.")
            if not np.isfinite(Xt).all():
                raise ValueError("Preprocessed X contains NaN or infinite values.")
            if np.any(Xt < 0.0) or np.any(Xt > 1.0):
                raise ValueError("Preprocessed X must contain values in [0, 1].")

            self.preprocessor_ = None
            self.n_transformed_features_ = Xt.shape[1]
            transformed_names = (
                list(self.feature_names)
                if self.feature_names is not None
                else [f"x{i + 1}" for i in range(Xt.shape[1])]
            )
            if len(transformed_names) != Xt.shape[1]:
                raise ValueError(
                    "When preprocessed=True, feature_names must match the number "
                    "of transformed columns."
                )
            self.transformed_feature_names_ = transformed_names
            self._configure_preprocessed_feature_metadata(Xt.shape[1], transformed_names)
        else:
            cat_cols, num_cols = self._resolve_categorical_and_numeric_columns(X)
            self.categorical_columns_ = cat_cols
            self.numeric_columns_ = num_cols

            transformers = []
            if len(num_cols) > 0:
                transformers.append(("num", MinMaxScaler(clip=True), num_cols))
            if len(cat_cols) > 0:
                transformers.append(("cat", self._make_one_hot_encoder(), cat_cols))
            if not transformers:
                raise ValueError("No input columns available for preprocessing.")

            self.preprocessor_ = ColumnTransformer(transformers, remainder="drop")
            Xt = np.ascontiguousarray(
                self.preprocessor_.fit_transform(X), dtype=np.float64
            )
            self.n_transformed_features_ = Xt.shape[1]
            self._build_transformed_feature_metadata_and_groups()

        best_rules, best_acc = self._beam_search(Xt, y_bin)
        self.rules_struct_ = best_rules
        self.train_acc_ = best_acc
        self.rule_stats_ = self._compute_rule_stats(Xt, y_bin, self.rules_struct_)
        self.rules_ = self._format_rules(self.rules_struct_, self.rule_stats_)
        self._compiled_rule_arrays_ = self._encode_single_ruleset(self.rules_struct_)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["rules_struct_", "_compiled_rule_arrays_"])
        self._check_X_no_return(X)
        if self.preprocessed:
            Xt = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        else:
            Xt = np.ascontiguousarray(self.preprocessor_.transform(X), dtype=np.float64)
        features, states, modifiers, lengths = self._compiled_rule_arrays_
        return _score_rules_fast(Xt, features, states, modifiers, lengths)

    def predict_proba(self, X):
        S = self.decision_function(X)
        p_pos = 1.0 - np.exp(2 * np.log(0.5) * S)
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X):
        pred_bin = (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
        return np.where(pred_bin == 1, self.positive_class_, self.negative_class_)

    def _beam_search(self, Xt, y_bin):
        Xt = np.ascontiguousarray(Xt, dtype=np.float64)
        y_bin = np.ascontiguousarray(y_bin, dtype=np.bool_)
        atomic_conditions = self._make_atomic_conditions()

        empty: RuleSet = tuple()
        empty_acc = float(self._evaluate_rules_batch(Xt, y_bin, [empty])[0])
        best_model, best_acc = empty, empty_acc
        beam = [(empty_acc, empty)]
        max_steps = (
            self.max_steps
            if self.max_steps is not None
            else self.max_rules
            * self._effective_max_rules_len_
            * self.max_literal_repetitions
        )
        eval_cache: Dict[RuleSet, float] = {empty: empty_acc}

        for step in range(max_steps):
            raw_unique = set()
            for _, model in beam:
                for candidate in self._expand_model(model, atomic_conditions):
                    canonical = self._canonicalize_rules(candidate)
                    if self._is_valid_ruleset(canonical):
                        raw_unique.add(canonical)

            if not raw_unique:
                break

            base_candidates = [m for m in raw_unique if m not in eval_cache]
            candidates_to_evaluate = []
            queued = set()

            for base in base_candidates:
                if base not in queued:
                    queued.add(base)
                    candidates_to_evaluate.append(base)

            if not candidates_to_evaluate:
                break

            accuracies = self._evaluate_rules_batch(Xt, y_bin, candidates_to_evaluate)
            scored = []
            for model, acc in zip(candidates_to_evaluate, accuracies):
                acc = float(acc)
                eval_cache[model] = acc
                scored.append((acc, model))
                if acc > best_acc:
                    best_acc, best_model = acc, model

            scored.sort(key=lambda item: item[0], reverse=True)
            beam = scored[: self.beam_width]
            if self.verbose:
                print(
                    f"[Beam step {step + 1}/{max_steps}] "
                    f"best_acc={best_acc:.6f}, level_best={beam[0][0]:.6f}, "
                    f"beam_size={len(beam)}, evaluated={len(candidates_to_evaluate)}"
                )

        return self._canonicalize_rules(best_model), best_acc

    def _make_atomic_conditions(self):
        conditions = []
        for j in range(self.n_transformed_features_):
            if j in self.continuous_feature_indices_:
                conditions.extend([Condition(j, "high"), Condition(j, "low")])
            else:
                conditions.extend([Condition(j, "present"), Condition(j, "absent")])
        return tuple(conditions)

    def _expand_model(self, model: RuleSet, atomic_conditions):
        candidates = []

        if len(model) < self.max_rules:
            for condition in atomic_conditions:
                candidates.append(model + ((condition,),))

        for r_idx, rule in enumerate(model):
            if self._rule_length(rule) >= self._effective_max_rules_len_:
                # Existing conditions may still be intensified without increasing
                # linguistic rule length.
                allow_new_feature = False
            else:
                allow_new_feature = True

            by_feature = {condition.feature: condition for condition in rule}

            # Add a new feature-level condition.
            if allow_new_feature:
                for condition in atomic_conditions:
                    if condition.feature not in by_feature:
                        new_rules = list(model)
                        new_rules[r_idx] = rule + (condition,)
                        candidates.append(tuple(new_rules))

            # Refine an existing continuous condition.
            for pos, existing in enumerate(rule):
                if existing.feature not in self.continuous_feature_indices_:
                    continue

                if (
                    existing.state in {"high", "low"}
                    and existing.modifier < self.max_literal_repetitions
                ):
                    refined = Condition(
                        existing.feature,
                        existing.state,
                        existing.modifier + 1,
                    )
                    new_rule = list(rule)
                    new_rule[pos] = refined
                    new_rules = list(model)
                    new_rules[r_idx] = tuple(new_rule)
                    candidates.append(tuple(new_rules))

                if existing.state in {"high", "low"} and existing.modifier == 1:
                    medium = Condition(existing.feature, "medium", 1)
                    new_rule = list(rule)
                    new_rule[pos] = medium
                    new_rules = list(model)
                    new_rules[r_idx] = tuple(new_rule)
                    candidates.append(tuple(new_rules))

        return candidates

    @staticmethod
    def _rule_length(rule: Rule) -> int:
        # Every Condition is already one user-visible linguistic condition.
        return len(rule)

    def _canonicalize_rules(self, rules: RuleSet) -> RuleSet:
        canonical_rules = []
        for rule in rules:
            clean = self._canonicalize_rule(rule)
            if clean is not None and self._rule_length(clean) <= self._effective_max_rules_len_:
                canonical_rules.append(clean)
        return tuple(sorted(set(canonical_rules), key=lambda r: (len(r), r)))

    def _canonicalize_rule(self, rule: Rule):
        if not rule:
            return None

        by_feature: Dict[int, Condition] = {}
        for condition in rule:
            previous = by_feature.get(condition.feature)
            if previous is None:
                by_feature[condition.feature] = condition
            elif previous != condition:
                # A rule should already hold one semantic condition per transformed
                # feature. Different conditions for the same feature are invalid.
                return None

        # Validate mutually exclusive one-hot groups and remove redundant
        # "absent other category" conditions implied by a present category.
        conditions = list(by_feature.values())
        for group in self.categorical_feature_groups_:
            group_conditions = [c for c in conditions if c.feature in group]
            present = [c for c in group_conditions if c.state == "present"]
            if len(present) > 1:
                return None
            if present:
                chosen = present[0]
                conditions = [
                    c
                    for c in conditions
                    if c.feature not in group or c == chosen
                ]

        return tuple(sorted(conditions))

    def _is_valid_ruleset(self, rules):
        return (
            bool(rules)
            and len(rules) <= self.max_rules
            and all(self._rule_length(rule) <= self._effective_max_rules_len_ for rule in rules)
        )

    def _encode_single_ruleset(self, rules):
        n_rules = len(rules)
        max_len = max((len(rule) for rule in rules), default=0)
        features = np.full((n_rules, max_len), -1, dtype=np.int64)
        states = np.zeros((n_rules, max_len), dtype=np.uint8)
        modifiers = np.ones((n_rules, max_len), dtype=np.uint8)
        lengths = np.zeros(n_rules, dtype=np.int64)

        for r, rule in enumerate(rules):
            lengths[r] = len(rule)
            for k, condition in enumerate(rule):
                features[r, k] = condition.feature
                states[r, k] = _STATE_TO_CODE[condition.state]
                modifiers[r, k] = condition.modifier
        return features, states, modifiers, lengths

    def _encode_ruleset_batch(self, candidates):
        n_candidates = len(candidates)
        features = np.full(
            (n_candidates, self.max_rules, self._effective_max_rules_len_), -1, dtype=np.int64
        )
        states = np.zeros_like(features, dtype=np.uint8)
        modifiers = np.ones_like(features, dtype=np.uint8)
        lengths = np.zeros((n_candidates, self.max_rules), dtype=np.int64)
        n_rules = np.zeros(n_candidates, dtype=np.int64)

        for c, rules in enumerate(candidates):
            n_rules[c] = len(rules)
            for r, rule in enumerate(rules):
                lengths[c, r] = len(rule)
                for k, condition in enumerate(rule):
                    features[c, r, k] = condition.feature
                    states[c, r, k] = _STATE_TO_CODE[condition.state]
                    modifiers[c, r, k] = condition.modifier
        return features, states, modifiers, lengths, n_rules

    def _evaluate_rules_batch(self, Xt, y_bin, candidates):
        features, states, modifiers, lengths, n_rules = self._encode_ruleset_batch(
            candidates
        )
        return _batch_accuracy_fast(
            np.ascontiguousarray(Xt, dtype=np.float64),
            np.ascontiguousarray(y_bin, dtype=np.bool_),
            features,
            states,
            modifiers,
            lengths,
            n_rules,
            self.threshold
        )

    def _score_rules(self, Xt, rules):
        features, states, modifiers, lengths = self._encode_single_ruleset(rules)
        return _score_rules_fast(
            np.ascontiguousarray(Xt, dtype=np.float64),
            features,
            states,
            modifiers,
            lengths,
        )

    def _compute_rule_stats(self, Xt, y_bin, rules):
        stats = []
        prior = max(float(np.mean(y_bin)), np.finfo(float).eps)
        for rule in rules:
            covered = np.ones(Xt.shape[0], dtype=bool)
            for condition in rule:
                x = Xt[:, condition.feature]
                if condition.state == "high":
                    cutoff = 0.5 ** (1.0 / condition.modifier)
                    covered &= x >= cutoff
                elif condition.state == "low":
                    cutoff = 1.0 - 0.5 ** (1.0 / condition.modifier)
                    covered &= x <= cutoff
                elif condition.state == "medium":
                    covered &= (x >= 0.25) & (x <= 0.75)
                elif condition.state == "present":
                    covered &= x >= 0.5
                else:
                    covered &= x < 0.5
            n_cov = int(covered.sum())
            n_pos = int(np.sum(y_bin[covered] == 1)) if n_cov else 0
            n_neg = n_cov - n_pos
            support = n_cov / Xt.shape[0]
            precision = n_pos / n_cov if n_cov else 0.0
            lift = precision / prior if n_cov else 0.0
            stats.append(RuleStats(support, lift, n_cov, n_pos, n_neg))
        return stats

    def _format_rules(self, rules, stats):
        pos_name = self.class_names[1] if self.class_names is not None else "1"
        neg_name = self.class_names[0] if self.class_names is not None else "0"
        formatted = []
        for rule, stat in zip(rules, stats):
            antecedent = " AND ".join(self._condition_to_text(c) for c in rule)
            formatted.append(
                f"IF {antecedent} THEN class is {pos_name} "
                f"| Lift: {stat.lift:.4f}; Support: {stat.support:.4f}; "
                f"Covered: {stat.n_covered}"
            )
        formatted.append(f"ELSE class is {neg_name}")
        return formatted

    def _condition_to_text(self, condition):
        meta = self.transformed_feature_metadata_[condition.feature]
        name = meta["name"]
        if condition.state in {"high", "low"}:
            modifier = {1: "", 2: "Very ", 3: "Extremely "}[condition.modifier]
            return f"{name} is {modifier}{condition.state.title()}"
        if condition.state == "medium":
            return f"{name} is Medium"
        category = meta["category"]
        return (
            f"{name} is {category}"
            if condition.state == "present"
            else f"{name} is not {category}"
        )

    def _validate_parameters(self):
        effective_len = (
            self.max_rule_len
            if self.max_rule_len is not None
            else self.max_rules_len
        )
        self._effective_max_rules_len_ = effective_len
        if self.max_rules < 1 or effective_len < 1 or self.beam_width < 1:
            raise ValueError("max_rules, max_rules_len, and beam_width must be >= 1.")
        if not 1 <= self.max_literal_repetitions <= 3:
            raise ValueError("max_literal_repetitions must be between 1 and 3.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if self.preprocessed and self.continuous_features is None:
            raise ValueError(
                "continuous_features is required when preprocessed=True. "
                "Use continuous_features='all' when appropriate."
            )

    def _configure_preprocessed_feature_metadata(self, n_features, names):
        if self.continuous_features == "all":
            continuous = set(range(n_features))
            groups = []
        else:
            continuous = self._resolve_index_spec(
                self.continuous_features, n_features, names, "continuous_features"
            )
            if self.categorical_feature_groups is None:
                if continuous != set(range(n_features)):
                    raise ValueError(
                        "categorical_feature_groups is required when preprocessed=True "
                        "and not all transformed features are continuous."
                    )
                groups = []
            else:
                groups = [frozenset(map(int, group)) for group in self.categorical_feature_groups]

        flat_groups = [j for group in groups for j in group]
        if len(flat_groups) != len(set(flat_groups)):
            raise ValueError("categorical_feature_groups must not overlap.")
        if any(j < 0 or j >= n_features for j in flat_groups):
            raise ValueError("categorical_feature_groups contains an invalid index.")
        categorical = set(flat_groups)
        if continuous & categorical:
            raise ValueError("Continuous and categorical transformed features overlap.")
        uncovered = set(range(n_features)) - continuous - categorical
        if uncovered:
            raise ValueError(
                f"Transformed feature indices are not described by metadata: {sorted(uncovered)}"
            )

        self.continuous_feature_indices_ = frozenset(continuous)
        self.categorical_feature_groups_ = tuple(groups)
        group_lookup = {}
        for gid, group in enumerate(groups):
            for j in group:
                group_lookup[j] = gid
        self.categorical_group_by_feature_ = group_lookup
        self.transformed_feature_metadata_ = []
        for j, name in enumerate(names):
            if j in continuous:
                self.transformed_feature_metadata_.append(
                    {"type": "numeric", "name": name, "category": None}
                )
            else:
                # With an external pipeline, feature_names should identify the
                # one-hot category, e.g. "vehicle_bicycle".
                self.transformed_feature_metadata_.append(
                    {"type": "categorical", "name": name, "category": "present"}
                )
        self.numeric_columns_ = sorted(continuous)
        self.categorical_columns_ = sorted(categorical)

    def _build_transformed_feature_metadata_and_groups(self):
        metadata = []
        continuous = []
        groups = []
        transformed_index = 0

        if "num" in self.preprocessor_.named_transformers_:
            for col in self.numeric_columns_:
                idx = self._column_to_index(col)
                metadata.append(
                    {"type": "numeric", "name": self._input_feature_names_[idx], "category": None}
                )
                continuous.append(transformed_index)
                transformed_index += 1

        if "cat" in self.preprocessor_.named_transformers_:
            encoder = self.preprocessor_.named_transformers_["cat"]
            for col, categories in zip(self.categorical_columns_, encoder.categories_):
                idx = self._column_to_index(col)
                base_name = self._input_feature_names_[idx]
                group = []
                for category in categories:
                    metadata.append(
                        {"type": "categorical", "name": base_name, "category": str(category)}
                    )
                    group.append(transformed_index)
                    transformed_index += 1
                groups.append(frozenset(group))

        self.transformed_feature_metadata_ = metadata
        self.transformed_feature_names_ = [
            item["name"]
            if item["type"] == "numeric"
            else f'{item["name"]}={item["category"]}'
            for item in metadata
        ]
        self.continuous_feature_indices_ = frozenset(continuous)
        self.categorical_feature_groups_ = tuple(groups)
        self.categorical_group_by_feature_ = {
            j: gid for gid, group in enumerate(groups) for j in group
        }

    @staticmethod
    def _resolve_index_spec(spec, n_features, names, parameter_name):
        if spec is None:
            return set()
        values = list(spec)
        if len(values) == n_features and all(isinstance(v, (bool, np.bool_)) for v in values):
            return {i for i, value in enumerate(values) if bool(value)}
        if all(isinstance(v, (int, np.integer)) for v in values):
            indices = {int(v) for v in values}
        elif all(isinstance(v, str) for v in values):
            mapping = {name: i for i, name in enumerate(names)}
            unknown = [value for value in values if value not in mapping]
            if unknown:
                raise ValueError(f"Unknown names in {parameter_name}: {unknown}")
            indices = {mapping[value] for value in values}
        else:
            raise ValueError(
                f"{parameter_name} must contain indices, names, or be a boolean mask."
            )
        if any(i < 0 or i >= n_features for i in indices):
            raise ValueError(f"{parameter_name} contains an invalid index.")
        return indices

    def _make_one_hot_encoder(self):
        try:
            return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover
            return OneHotEncoder(sparse=False, handle_unknown="ignore")

    def _resolve_input_feature_names(self, X):
        if self.feature_names is not None:
            names = list(self.feature_names)
        elif pd is not None and isinstance(X, pd.DataFrame):
            names = list(map(str, X.columns))
        else:
            names = [f"x{i + 1}" for i in range(X.shape[1])]
        if len(names) != X.shape[1]:
            raise ValueError("feature_names length must match the number of input columns.")
        return names

    def _resolve_categorical_and_numeric_columns(self, X):
        n = X.shape[1]
        names = self._input_feature_names_
        cat = self.categorical_features
        if cat is None:
            if pd is not None and isinstance(X, pd.DataFrame):
                cat_indices = [
                    i
                    for i, col in enumerate(X.columns)
                    if str(X[col].dtype) in ("object", "category", "bool")
                ]
            else:
                cat_indices = []
        else:
            cat_indices = sorted(
                self._resolve_index_spec(cat, n, names, "categorical_features")
            )
        num_indices = [i for i in range(n) if i not in set(cat_indices)]
        if pd is not None and isinstance(X, pd.DataFrame) and self.feature_names is None:
            return [X.columns[i] for i in cat_indices], [X.columns[i] for i in num_indices]
        return cat_indices, num_indices

    def _column_to_index(self, col):
        if isinstance(col, (int, np.integer)):
            return int(col)
        return self._input_feature_names_.index(str(col))

    def _check_X_y_no_return(self, X, y):
        try:
            check_X_y(X, y, dtype=None, force_all_finite=True)
        except TypeError:
            check_X_y(X, y, dtype=None, ensure_all_finite=True)

    def _check_X_no_return(self, X):
        try:
            check_array(X, dtype=None, force_all_finite=True)
        except TypeError:
            check_array(X, dtype=None, ensure_all_finite=True)
