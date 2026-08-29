# ============================================================
# The code was written in cooperation with GPT-5.6 Think deeper
# ============================================================

"""Random-search classifier based on interpretable fuzzy rule sets.

This module implements a binary classifier that constructs a collection of
random fuzzy RuleSets, evaluates their training accuracy, and retains the
best candidate. Accuracy ties are resolved in favor of simpler models.

A RuleSet is a disjunction of rules. Each rule is a conjunction of fuzzy
conditions. The activation of a rule is calculated as the product of its
condition-membership values, while the score of a RuleSet is the sum of its
rule activations.

Continuous transformed features support the linguistic states ``high``
and ``low``. High and low conditions may additionally use linguistic modifiers 
represented by powers from one to three. Categorical one-hot encoded features 
support the states ``present`` and ``absent``.

Candidate RuleSets are generated, canonicalized, validated, and deduplicated
inside Numba-compiled functions. The number of rules and individual rule
lengths can independently follow either a discrete uniform distribution or
an exponential/log-uniform distribution. Candidate evaluation is parallelized
over RuleSets with Numba.

The estimator can preprocess raw numerical and categorical data internally,
or consume an externally preprocessed dense matrix whose values lie in the
closed interval [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import time
import warnings

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

_STATE_TO_CODE = {"high": 0, "low": 1, "present": 2, "absent": 3}


@dataclass(frozen=True, order=True)
class Condition:
    """Represent one fuzzy condition in a rule.

    A condition applies a linguistic state to one transformed input feature.
    Continuous features support ``high`` and ``low``.
    Categorical one-hot encoded features support ``present`` and ``absent``
    states.
    
    The ``modifier`` controls the power used for high and low memberships:
    
    - ``modifier=1``: High or Low,
    - ``modifier=2``: Very High or Very Low,
    - ``modifier=3``: Extremely High or Extremely Low.
    
    The ``present`` and ``absent`` states do not support modifiers
    other than one.
    
    Parameters
    ----------
    feature : int
        Zero-based index of the transformed feature to which the condition
        applies.
    
    state : {"high", "low", "present", "absent"}
        Linguistic state of the condition.
    
    modifier : int, default=1
        Membership-function exponent. Values from one to three are allowed.
        Values greater than one are valid only for ``high`` and ``low``.
    
    Attributes
    ----------
    feature : int
        Index of the transformed feature.
    
    state : str
        Linguistic condition state.
    
    modifier : int
        Membership-function exponent.
    
    Notes
    -----
    The dataclass is immutable and orderable. Its ordering follows the tuple
    ``(feature, state, modifier)`` and can therefore be used when constructing
    canonical representations of rules.
    """
    feature: int
    state: str
    modifier: int = 1

    def __post_init__(self):
        """Validate the state and modifier of a newly created condition.

        Raises
        ------
        ValueError
            If ``state`` is not one of the supported linguistic states.
        
        ValueError
            If ``modifier`` is outside the inclusive interval [1, 3].
        
        ValueError
            If a modifier other than one is used with ``present``
            or ``absent``.
        """
        if self.state not in _STATE_TO_CODE:
            raise ValueError(f"Unknown state: {self.state!r}")
        if not 1 <= self.modifier <= 3:
            raise ValueError("modifier must be in [1, 3]")
        if self.state in {"present", "absent"} and self.modifier != 1:
            raise ValueError(f"{self.state!r} only supports modifier=1")


Rule = Tuple[Condition, ...]
RuleSet = Tuple[Rule, ...]


@dataclass(frozen=True)
class RuleStats:
    """Store descriptive statistics for one fitted fuzzy rule.

    Parameters
    ----------
    support : float
        Fraction of training observations covered by the rule.
    
    lift : float
        Ratio between the positive-class precision among covered observations
        and the positive-class prior probability.
    
    n_covered : int
        Number of training observations covered by the rule.
    
    n_positive_covered : int
        Number of covered observations belonging to the positive class.
    
    n_negative_covered : int
        Number of covered observations belonging to the negative class.
    
    Attributes
    ----------
    support : float
        Relative rule coverage.
    
    lift : float
        Positive-class lift of the rule.
    
    n_covered : int
        Total number of covered observations.
    
    n_positive_covered : int
        Number of covered positive observations.
    
    n_negative_covered : int
        Number of covered negative observations.
    """
    support: float
    lift: float
    n_covered: int
    n_positive_covered: int
    n_negative_covered: int

                   
@njit(cache=True, fastmath=False, parallel=False)
def _score_rules_fast(X, features, states, modifiers, lengths):
    """Calculate raw RuleSet scores for all observations.

    The activation of each rule is the product of its condition-membership
    values. The RuleSet score is the sum of the activations of all rules.
    
    For a transformed feature value ``x``, the supported memberships are:
    
    - High: ``x ** modifier``,
    - Low: ``(1 - x) ** modifier``,
    - Present: ``x``,
    - Absent: ``1 - x``.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dense transformed input matrix. Values are expected to lie in [0, 1].
    
    features : ndarray of shape (n_rules, max_rule_length)
        Transformed-feature indices used by the conditions of each rule.
        Only the first ``lengths[r]`` entries of rule ``r`` are active.
    
    states : ndarray of uint8, shape (n_rules, max_rule_length)
        Integer state codes for the encoded conditions:
    
        - 0: high,
        - 1: low,
        - 2: present,
        - 3: absent.
    
    modifiers : ndarray of uint8, shape (n_rules, max_rule_length)
        Membership exponents. High and low conditions support values from one
        to three. Other states use one.
    
    lengths : ndarray of shape (n_rules,)
        Number of active conditions in every rule.
    
    Returns
    -------
    scores : ndarray of float64, shape (n_samples,)
        Sum of fuzzy rule activations for each observation.
    
    Notes
    -----
    This function scores one already selected RuleSet and is used by
    ``decision_function``. It does not apply the probability transformation or
    classification threshold.
    """
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
                    value = x
                else:
                    value = 1.0 - x
                activation *= value
            total += activation
        out[i] = total
    return out


@njit(cache=True, fastmath=False, parallel=True)
def _batch_confusion(X, y, features, states, modifiers, lengths, n_rules, threshold):
    """Calculate confusion counts for a batch of encoded RuleSets.

    Candidate RuleSets are evaluated independently and in parallel. For every
    observation, the raw fuzzy score is converted to a positive-class
    probability using

    ``p = 1 - exp(2 * log(0.5) * score)``.

    The predicted class is positive when ``p >= threshold``.

    Returns
    -------
    true_positives : ndarray of int64, shape (n_candidates,)
        Number of correctly predicted positive observations.

    true_negatives : ndarray of int64, shape (n_candidates,)
        Number of correctly predicted negative observations.

    false_positives : ndarray of int64, shape (n_candidates,)
        Number of negative observations predicted as positive.

    false_negatives : ndarray of int64, shape (n_candidates,)
        Number of positive observations predicted as negative.

    """
    n_candidates = features.shape[0]

    true_positives = np.zeros(n_candidates, dtype=np.int64)
    true_negatives = np.zeros(n_candidates, dtype=np.int64)
    false_positives = np.zeros(n_candidates, dtype=np.int64)
    false_negatives = np.zeros(n_candidates, dtype=np.int64)
    
    log_quarter = 2.0 * np.log(0.5)
    
    for c in prange(n_candidates):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
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
                        value = x
                    else:
                        value = 1.0 - x
                    activation *= value
                score += activation
            p = 1.0 - np.exp(log_quarter * score)
            y_pred = p >= threshold
            
            if y_pred:
                if y[i]:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i]:
                    fn += 1
                else:
                    tn += 1
                    
        true_positives[c] = tp
        true_negatives[c] = tn
        false_positives[c] = fp
        false_negatives[c] = fn
    return true_positives, true_negatives, false_positives, false_negatives


@njit(cache=True, fastmath=False, parallel=True)
def _batch_confusion_threshold_half(X, y, features, states, modifiers, lengths, n_rules):
    """Calculate confusion counts using the specialized threshold of 0.5.

    This function is equivalent to ``_batch_confusion`` when the probability
    threshold equals 0.5. Under the probability transformation
    
    ``p = 1 - exp(2 * log(0.5) * score)``,
    
    the condition ``p >= 0.5`` is equivalent to ``score >= 0.5``. The expensive
    probability transformation can therefore be omitted.
    
    Because all rule activations are non-negative, the RuleSet score can only
    increase as additional rules are evaluated. Rule evaluation for an
    observation is stopped as soon as the accumulated score reaches 0.5.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dense transformed training matrix with values in [0, 1].
    
    y : ndarray of bool, shape (n_samples,)
        Binary training labels. ``True`` denotes the positive class.
    
    features : ndarray of shape (n_candidates, max_rules, max_rule_length)
        Encoded transformed-feature indices for all candidate RuleSets.
    
    states : ndarray of uint8, shape (n_candidates, max_rules, max_rule_length)
        Encoded condition-state codes.
    
    modifiers : ndarray of uint8, shape (n_candidates, max_rules, max_rule_length)
        Encoded membership exponents.
    
    lengths : ndarray of shape (n_candidates, max_rules)
        Number of active conditions in every rule.
    
    n_rules : ndarray of shape (n_candidates,)
        Number of active rules in every candidate.

    Returns
    -------
    true_positives : ndarray of int64, shape (n_candidates,)
        Number of correctly predicted positive observations.

    true_negatives : ndarray of int64, shape (n_candidates,)
        Number of correctly predicted negative observations.

    false_positives : ndarray of int64, shape (n_candidates,)
        Number of negative observations predicted as positive.

    false_negatives : ndarray of int64, shape (n_candidates,)
        Number of positive observations predicted as negative.

    """
    n_candidates = features.shape[0]

    true_positives = np.zeros(n_candidates, dtype=np.int64)
    true_negatives = np.zeros(n_candidates, dtype=np.int64)
    false_positives = np.zeros(n_candidates, dtype=np.int64)
    false_negatives = np.zeros(n_candidates, dtype=np.int64)
    
    for c in prange(n_candidates):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
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
                        value = x
                    else:
                        value = 1.0 - x
                    activation *= value
                score += activation
                if score >= 0.5:
                    break
            y_pred = score >= 0.5
            
            if y_pred:
                if y[i]:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i]:
                    fn += 1
                else:
                    tn += 1
                    
        true_positives[c] = tp
        true_negatives[c] = tn
        false_positives[c] = fp
        false_negatives[c] = fn
    return true_positives, true_negatives, false_positives, false_negatives

@njit(
    cache=True,
    fastmath=False,
    parallel=True,
)
def _batch_mcc_from_confusion(true_positives, true_negatives, false_positives, false_negatives):
    """Calculate MCC values from candidate confusion-matrix counts.

    Parameters
    ----------
    true_positives : ndarray of int64, shape (n_candidates,)
        True-positive counts for all candidate RuleSets.

    true_negatives : ndarray of int64, shape (n_candidates,)
        True-negative counts for all candidate RuleSets.

    false_positives : ndarray of int64, shape (n_candidates,)
        False-positive counts for all candidate RuleSets.

    false_negatives : ndarray of int64, shape (n_candidates,)
        False-negative counts for all candidate RuleSets.

    Returns
    -------
    objective_values : ndarray of float64, shape (n_candidates,)
        Matthews correlation coefficient for every candidate. A candidate
        receives zero when the MCC denominator is zero.

    Notes
    -----
    The candidate loop is parallelized with ``numba.prange``.

    Confusion counts are converted to floating-point values before
    multiplication. This prevents integer overflow when evaluating large
    training sets.
    """
    n_candidates = true_positives.shape[0]

    objective_values = np.empty(n_candidates, dtype=np.float64)

    for candidate_index in prange(n_candidates):
        tp = float(true_positives[candidate_index])
        tn = float(true_negatives[candidate_index])
        fp = float(false_positives[candidate_index])
        fn = float(false_negatives[candidate_index])

        numerator = tp * tn - fp * fn

        denominator_squared = (
            (tp + fp)
            * (tp + fn)
            * (tn + fp)
            * (tn + fn)
        )

        if denominator_squared > 0.0:
            objective_values[candidate_index] = numerator / np.sqrt(denominator_squared)
        else:
            objective_values[candidate_index] = 0.0

    return objective_values

@njit(cache=True, fastmath=False, inline="always", forceinline=True)
def _sample_structure_size(max_value, sampling_type):
    """Sample a RuleSet size or an individual rule length.

    Parameters
    ----------
    max_value : int
        Maximum allowed integer. The returned value belongs to the inclusive
        interval [1, max_value].
    
    sampling_type : {1, 2}
        Sampling strategy:
    
        - 1: discrete uniform sampling from ``{1, ..., max_value}``;
        - 2: exponential/log-uniform sampling calculated as
          ``floor(2 ** U(0, log2(max_value + 1)))``.
    
    Returns
    -------
    value : int
        Sampled integer in the inclusive interval [1, max_value].
    
    Notes
    -----
    The exponential strategy assigns greater probability to smaller values while
    retaining non-zero probability for every value up to ``max_value``. Using
    ``log2(max_value + 1)`` rather than ``log2(max_value)`` ensures that
    ``max_value`` itself can be sampled with non-zero probability.
    
    If ``max_value`` equals one, the function returns one without drawing a
    random number.
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
def _hash_candidate_fast(features, states, modifiers, lengths, n_rules):
    """Calculate a deterministic 64-bit hash of a canonical RuleSet.

    The hash incorporates the number of rules, every active rule length, and the
    feature, state, and modifier of every active condition. Padding entries are
    ignored.
    
    Parameters
    ----------
    features : ndarray of shape (max_rules, max_rule_length)
        Canonically ordered transformed-feature indices of one candidate.
    
    states : ndarray of uint8, shape (max_rules, max_rule_length)
        Canonically ordered condition-state codes.
    
    modifiers : ndarray of uint8, shape (max_rules, max_rule_length)
        Canonically ordered condition modifiers.
    
    lengths : ndarray of shape (max_rules,)
        Number of active conditions in every rule.
    
    n_rules : int
        Number of active rules in the candidate.
    
    Returns
    -------
    hash_value : numpy.uint64
        Deterministic 64-bit hash of the active RuleSet representation.
    
    Notes
    -----
    The function uses an FNV-style sequence of XOR and multiplication operations.
    The hash is used to locate potential duplicates in an open-addressing hash
    table. Hash equality alone is not treated as proof of candidate equality:
    matching hashes are followed by an exact structural comparison.
    """
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
    candidate_n_rules
):
    """Compare a temporary candidate with an accepted stored candidate.

    Only active rules and active conditions are compared. Padding in the
    fixed-shape arrays is ignored.
    
    Parameters
    ----------
    stored_features : ndarray of shape (n_candidates, max_rules, max_rule_length)
        Feature-index arrays of previously accepted candidates.
    
    stored_states : ndarray of uint8, shape (n_candidates, max_rules, max_rule_length)
        State-code arrays of previously accepted candidates.
    
    stored_modifiers : ndarray of uint8, shape (n_candidates, max_rules, max_rule_length)
        Modifier arrays of previously accepted candidates.
    
    stored_lengths : ndarray of shape (n_candidates, max_rules)
        Rule lengths of previously accepted candidates.
    
    stored_n_rules : ndarray of shape (n_candidates,)
        Number of rules in previously accepted candidates.
    
    stored_index : int
        Index of the accepted candidate to compare.
    
    candidate_features : ndarray of shape (max_rules, max_rule_length)
        Feature-index array of the temporary candidate.
    
    candidate_states : ndarray of uint8, shape (max_rules, max_rule_length)
        State-code array of the temporary candidate.
    
    candidate_modifiers : ndarray of uint8, shape (max_rules, max_rule_length)
        Modifier array of the temporary candidate.
    
    candidate_lengths : ndarray of shape (max_rules,)
        Rule lengths of the temporary candidate.
    
    candidate_n_rules : int
        Number of active rules in the temporary candidate.
    
    Returns
    -------
    equal : bool
        ``True`` if both candidates have exactly the same canonical active
        representation; otherwise ``False``.
    
    Notes
    -----
    This exact comparison resolves possible collisions in the 64-bit candidate
    hash.
    """
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
    """Validate and canonicalize one encoded candidate RuleSet in-place.

    The function validates individual rules, sorts conditions within each rule,
    sorts complete rules within the RuleSet, and rejects repeated rules.
    
    A candidate is rejected if a rule contains the same transformed feature more
    than once. For features belonging to one one-hot encoded categorical group,
    a rule is also rejected if two conditions from the group are present and at
    least one of them uses the ``present`` state. Multiple distinct ``absent``
    conditions from the same categorical group remain valid.
    
    Conditions are sorted lexicographically by
    
    ``(feature, state, modifier)``.
    
    Rules are sorted first by rule length and then lexicographically by their
    canonical conditions.
    
    Parameters
    ----------
    features : ndarray of shape (max_rules, max_rule_length)
        Feature indices of one temporary candidate. The array is modified
        in-place.
    
    states : ndarray of uint8, shape (max_rules, max_rule_length)
        Condition-state codes. The array is modified in-place.
    
    modifiers : ndarray of uint8, shape (max_rules, max_rule_length)
        Condition modifiers. The array is modified in-place.
    
    lengths : ndarray of shape (max_rules,)
        Number of active conditions in every rule. Active rule lengths may be
        reordered in-place.
    
    n_rules : int
        Number of active rules in the temporary candidate.
    
    group_ids : ndarray of shape (n_features,)
        Categorical group identifier for every transformed feature. A value of
        ``-1`` denotes a continuous feature. Non-negative values identify
        columns belonging to the same original one-hot encoded variable.
    
    Returns
    -------
    valid : bool
        ``True`` if the candidate is structurally valid after canonicalization;
        otherwise ``False``.
    
    Notes
    -----
    Candidates are rejected rather than shortened. This preserves exactly the
    sampled number of rules and sampled rule lengths.
    
    Canonicalization makes logically order-independent representations identical:
    changing the order of conditions inside a conjunction or the order of rules
    inside a RuleSet does not produce a distinct candidate.
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
    sampling_type_length
):
    """Generate unique valid canonical RuleSets in one Numba-compiled call.

    Candidate RuleSets are generated one at a time. Each candidate is immediately
    canonicalized, structurally validated, hashed, and checked for duplication.
    Only valid unique candidates are written to the preallocated output arrays.
    
    The number of rules and individual rule lengths are sampled independently.
    Both dimensions may use either a discrete uniform distribution or an
    exponential/log-uniform distribution favoring smaller values.
    
    Parameters
    ----------
    seed : int
        Seed used to initialize the NumPy random generator inside the compiled
        function.
    
    n_candidates : int
        Target number of valid unique RuleSets to generate.
    
    max_sampling_attempts : int
        Maximum number of raw candidate-generation attempts. The returned
        collection may contain fewer than ``n_candidates`` candidates if this
        limit is reached.
    
    max_rules : int
        Maximum number of rules allowed in one RuleSet.
    
    max_rule_length : int
        Maximum number of conditions allowed in one rule.
    
    max_modifier : int
        Maximum exponent allowed for high and low memberships. Supported values
        are from one to three.
    
    continuous_mask : ndarray of bool, shape (n_features,)
        Boolean mask indicating which transformed features are continuous.
        ``False`` entries denote one-hot encoded categorical columns.
    
    group_ids : ndarray of int64, shape (n_features,)
        Categorical group identifier for every transformed feature. Continuous
        features use ``-1``. Columns derived from the same original categorical
        variable share a non-negative identifier.
    
    sampling_type_number : {1, 2}
        Sampling strategy for the number of rules:
    
        - 1: discrete uniform;
        - 2: exponential/log-uniform.
    
    sampling_type_length : {1, 2}
        Sampling strategy for individual rule lengths:
    
        - 1: discrete uniform;
        - 2: exponential/log-uniform.
    
    Returns
    -------
    features : ndarray of int64, shape (n_generated, max_rules, max_rule_length)
        Feature indices of accepted canonical candidates.
    
    states : ndarray of uint8, shape (n_generated, max_rules, max_rule_length)
        State codes of accepted canonical candidates.
    
    modifiers : ndarray of uint8, shape (n_generated, max_rules, max_rule_length)
        Condition modifiers of accepted canonical candidates.
    
    lengths : ndarray of int64, shape (n_generated, max_rules)
        Number of active conditions in each accepted rule.
    
    n_rules : ndarray of int64, shape (n_generated,)
        Number of active rules in each accepted candidate.
    
    attempts : int
        Total number of raw RuleSets generated before reaching the requested
        candidate count or the attempt limit.
    
    invalid_count : int
        Number of raw candidates rejected because of structural invalidity or
        internal duplicate rules.
    
    duplicate_count : int
        Number of valid canonical candidates rejected because an identical
        RuleSet had already been accepted.
    
    Notes
    -----
    The function uses an open-addressing hash table with linear probing. The table
    has a power-of-two capacity at least twice ``n_candidates``, keeping the
    maximum load factor at or below 0.5.
    
    Hash collisions are handled safely. Candidates with equal hashes are compared
    structurally before being classified as duplicates.
    
    Unused locations in the fixed-shape output arrays are represented by padding:
    feature indices use ``-1``, state codes use zero, modifiers use one, and rule
    lengths use zero.
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
                    candidate_states[rule_index,literal_index] = np.random.randint(0, 2)
                    candidate_modifiers[rule_index, literal_index] = np.random.randint(1, max_modifier + 1)

                else:
                    candidate_states[rule_index, literal_index] = np.random.randint(2, 4)

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
    """Binary fuzzy-rule classifier trained through independent random search.

    The estimator generates a collection of random candidate RuleSets, evaluates
    their training accuracy, and retains the best candidate. An Accuracy tie is
    resolved lexicographically in favor of:
    
    1. fewer total conditions,
    2. fewer rules,
    3. a smaller sum of active condition modifiers.
    
    A RuleSet is interpreted as a disjunction of fuzzy rules. Each rule is a
    conjunction whose activation is the product of its condition memberships.
    The raw RuleSet score is the sum of all rule activations.
    
    For continuous features, the supported fuzzy membership values are:
    
    - High: ``x ** modifier``,
    - Low: ``(1 - x) ** modifier``.
    
    For one-hot encoded categorical features:
    
    - Present: ``x``,
    - Absent: ``1 - x``.
    
    Candidate RuleSets are generated, canonicalized, validated, and deduplicated
    inside a Numba-compiled function. The number of rules and individual rule
    lengths can independently follow either a discrete uniform or an
    exponential/log-uniform distribution.
    
    Parameters
    ----------
    max_rules : int, default=6
        Maximum number of rules in one candidate RuleSet.
    
    max_rules_len : int, default=6
        Maximum number of conditions in one rule. This parameter is used unless
        the deprecated alias ``max_rule_len`` is provided.
    
    max_rule_len : int or None, default=None
        Optional alias overriding ``max_rules_len``. Retained for API
        compatibility.
    
    max_literal_repetitions : int, default=3
        Maximum exponent allowed for high and low memberships:
    
        - 1: High or Low,
        - 2: up to Very High or Very Low,
        - 3: up to Extremely High or Extremely Low.
    
    threshold : float, default=0.5
        Positive-class probability threshold. If the value equals 0.5, a
        specialized evaluation kernel avoids the probability transformation and
        may stop evaluating rules after the raw score reaches 0.5.
    
    n_candidates : int, default=20000
        Target number of valid unique candidate RuleSets.
    
    max_sampling_attempts : int, default=1000000
        Maximum number of raw candidate-generation attempts. Fewer than
        ``n_candidates`` candidates may be evaluated if the limit is reached.
    
    sampling_type_number : {1, 2}, default=1
        Distribution used to sample the number of rules:
    
        - 1: discrete uniform;
        - 2: exponential/log-uniform, favoring smaller RuleSets.
    
    sampling_type_length : {1, 2}, default=1
        Distribution used to sample individual rule lengths:
    
        - 1: discrete uniform;
        - 2: exponential/log-uniform, favoring shorter rules.
    
    class_names : sequence of str or None, default=None
        Optional display names of the negative and positive classes, in that
        order. The names affect only formatted rule descriptions.
    
    feature_names : sequence of str or None, default=None
        Optional feature names. When ``preprocessed=True``, the sequence must
        describe transformed columns and have the same length as the input
        matrix. Otherwise, original DataFrame column names are used when
        available.
    
    categorical_features : sequence of int, bool, or str, or None, default=None
        Original input features to treat as categorical when preprocessing is
        performed internally. The specification may contain indices, names, or
        a Boolean mask.
    
    continuous_features : {"all"} or sequence of int, bool, or str, or None
        Transformed continuous-feature specification used when
        ``preprocessed=True``. Use ``"all"`` if every transformed input column
        is continuous.
    
    categorical_feature_groups : sequence of sequences of int or None
        Groups of transformed one-hot encoded columns. Each inner sequence
        identifies columns derived from one original categorical feature.
        Required in preprocessed mode whenever some transformed columns are not
        continuous.
    
    preprocessed : bool, default=False
        If ``True``, ``X`` is interpreted as a dense, finite, already transformed
        matrix with values in ``[0, 1]``. An optional quantile transformation may
        still be fitted internally to columns identified by
        ``continuous_features``.
    
        If ``False``, categorical features are one-hot encoded internally.
        Numerical features are processed with ``MinMaxScaler`` when
        ``quantile_transform=None`` or with the configured quantile-based
        transformation otherwise.

    quantile_transform : {None, "uniform", "normal"}, default=None
        Optional quantile-based transformation applied only to continuous
        features.
    
        If ``None``, continuous features are processed using the standard
        min-max scaling employed by the classifier.
    
        If ``"uniform"``, each continuous feature is independently mapped to an
        approximately uniform marginal distribution on ``[0, 1]``.
    
        If ``"normal"``, each continuous feature is first mapped to an
        approximately normal marginal distribution and subsequently rescaled to
        ``[0, 1]`` with ``MinMaxScaler(clip=True)``.
    
        One-hot-encoded categorical columns are never quantile-transformed. When
        ``preprocessed=True``, the transformation is applied to columns identified
        by ``continuous_features``. When ``preprocessed=False``, it is applied to
        numerical columns identified by the internal preprocessing logic.

    objective_fun : callable or None, default=None
        Optional function used to evaluate candidate RuleSets from their binary
        confusion-matrix counts. The callable must have the signature
    
        ``objective_fun(tp, tn, fp, fn) -> float``
    
        where ``tp``, ``tn``, ``fp``, and ``fn`` are non-negative integer counts.
        Higher returned values are interpreted as better candidate performance.
    
        If ``None``, the Matthews correlation coefficient is calculated by a
        dedicated Numba-compiled batch function. Candidates for which the MCC
        denominator is zero receive an objective value of zero.
    
        A custom callable is executed once per candidate outside the
        Numba-compiled evaluation kernel and does not need to be Numba-compatible.
        It must return one finite real scalar value for every candidate.
    
        The objective must depend only on thresholded binary predictions. Metrics
        requiring continuous decision scores or probabilities, such as ROC AUC,
        are not supported by this interface.
    
    random_state : int or None, default=None
        Seed controlling candidate generation. A fixed integer makes sampling
        reproducible. ``None`` requests non-deterministic seeding.
    
    verbose : int, default=0
        Verbosity level reserved for progress and diagnostic output.
    
    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Original negative and positive class labels.
    
    negative_class_ : object
        Original class label treated as negative.
    
    positive_class_ : object
        Original class label treated as positive.
    
    class_prior_positive_ : float
        Fraction of positive observations in the training target.
    
    n_features_in_ : int
        Number of original input features.
    
    n_transformed_features_ : int
        Number of columns after preprocessing.
    
    preprocessor_ : ColumnTransformer or None
        Fitted internal preprocessor used when ``preprocessed=False``. It contains
        numerical scaling or quantile transformation and categorical one-hot
        encoding. ``None`` when ``preprocessed=True``.
    
    transformed_feature_names_ : list of str
        Names of transformed input columns.
    
    continuous_feature_indices_ : frozenset of int
        Indices of transformed continuous features.
    
    categorical_feature_groups_ : tuple of frozenset
        Groups of transformed one-hot encoded columns.
    
    transformed_feature_metadata_ : list of dict
        Metadata used to format conditions as readable text.
    
    rules_struct_ : tuple of tuple of Condition
        Structured representation of the selected RuleSet.
    
    rules_ : list of str
        Human-readable selected rules, including support and lift statistics,
        followed by the default negative-class rule.
    
    rule_stats_ : list of RuleStats
        Descriptive statistics for selected positive-class rules.
    
    sampling_time_ : float
        Candidate-generation time in seconds.
    
    evaluation_time_ : float
        Candidate-evaluation time in seconds.
    
    n_sampling_attempts_ : int
        Number of raw candidate-generation attempts.
    
    n_unique_candidates_ : int
        Number of valid unique candidates generated and evaluated.
    
    n_invalid_candidates_ : int
        Number of structurally invalid candidates rejected during generation.
    
    n_duplicate_candidates_ : int
        Number of valid candidates rejected as duplicates.
    
    valid_sampling_attempts_ : int
        Number of sampling attempts that produced structurally valid candidates,
        including both unique candidates and duplicates.
    
    duplicate_rate_ : float
        Fraction of structurally valid candidates that were duplicates.
    
    invalid_rate_ : float
        Fraction of all sampling attempts rejected as structurally invalid.

    quantile_transformer_ : QuantileTransformer, Pipeline, or None
        Fitted quantile transformation used for continuous columns when
        ``preprocessed=True``. For ``quantile_transform="normal"``, the object is
        a pipeline containing a quantile transformer followed by min-max scaling.
        ``None`` when quantile transformation is disabled or is already included
        in ``preprocessor_``.
    
    quantile_feature_indices_ : ndarray of int
        Indices of transformed continuous columns to which
        ``quantile_transformer_`` is applied in preprocessed mode.

    objective_value_ : float
        Training objective value of the selected RuleSet. When
        ``objective_fun=None``, this is the Matthews correlation coefficient.
    
    Notes
    -----
    The classifier supports binary classification only. The positive class is the
    second element of ``numpy.unique(y)``.
    
    Candidate generation is stochastic, but candidate evaluation and tie-breaking
    are deterministic for a fixed generated candidate collection.

    When quantile transformation is enabled, fuzzy states describe the relative
    position of a value within the empirical marginal distribution of a feature
    rather than its position on the original measurement scale.
    
    The quantile transformer is fitted exclusively on the data passed to
    ``fit``. In cross-validation, it must therefore be fitted separately within
    each training fold.
    
    The estimator follows the scikit-learn estimator interface and supports
    ``fit``, ``predict``, ``predict_proba``, and ``decision_function``.
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
        quantile_transform=None,
        objective_fun=None,
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
        self.quantile_transform = quantile_transform
        self.objective_fun = objective_fun
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Generate candidate RuleSets and fit the best fuzzy-rule classifier.
        
        The method validates the target, preprocesses the input if necessary,
        generates valid unique random RuleSets, evaluates every candidate on the
        training data, and selects the candidate with the highest training Accuracy.
        Accuracy ties are resolved in favor of lower model complexity.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
            If ``preprocessed=True``, ``X`` must be a dense numerical matrix
            containing only finite values in [0, 1].
        
            If ``preprocessed=False``, ``X`` may contain original numerical and
            categorical features. Numerical columns are scaled to [0, 1], and
            categorical columns are one-hot encoded.
        
        y : array-like of shape (n_samples,)
            Binary target labels.
        
        Returns
        -------
        self : RandomFuzzyRulesClassifier
            Fitted estimator.
        
        Raises
        ------
        ValueError
            If estimator parameters are invalid.
        
        ValueError
            If the target does not contain exactly two classes.
        
        ValueError
            If ``class_names`` is supplied with a length other than two.
        
        ValueError
            If preprocessed input is not two-dimensional, contains non-finite
            values, or contains values outside [0, 1].
        
        ValueError
            If transformed feature metadata is inconsistent with the input matrix.
        
        RuntimeError
            If no valid candidate RuleSet can be generated.
        
        Notes
        -----
        The second value returned by ``numpy.unique(y)`` is treated as the positive
        class.
        
        When ``threshold == 0.5``, candidate evaluation uses a specialized kernel
        that compares the raw fuzzy score directly with 0.5 and stops evaluating
        additional rules as soon as a positive prediction is guaranteed.
        """
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
            
            if (Xt.ndim != 2 or not np.isfinite(Xt).all()):
                raise ValueError("Preprocessed X must be a finite two-dimensional array.")
            if np.any(Xt < 0.0) or np.any(Xt > 1.0):
                raise ValueError("Preprocessed X values must lie in [0, 1].")
                
            self.n_transformed_features_ = Xt.shape[1]
        
            names = (list(self.feature_names) if self.feature_names is not None else [f"x{i + 1}" for i in range(Xt.shape[1])])
        
            if len(names) != Xt.shape[1]:
                raise ValueError("feature_names must match transformed columns.")
        
            self.transformed_feature_names_ = names
        
            self._configure_preprocessed_metadata(Xt.shape[1], names)
        
            self.quantile_feature_indices_ = np.asarray(sorted(self.continuous_feature_indices_), dtype=np.int64)
        
            if self.quantile_transform is not None and self.quantile_feature_indices_.size > 0:
                self.quantile_transformer_ = self._make_quantile_transformer(n_samples=Xt.shape[0])
        
                Xt = np.array(Xt, dtype=np.float64, order="C", copy=True)
        
                Xt[:, self.quantile_feature_indices_] = (
                    self.quantile_transformer_.fit_transform(Xt[:, self.quantile_feature_indices_])
                )
            else:
                self.quantile_transformer_ = None
        
            self.preprocessor_ = None
        
            Xt = np.ascontiguousarray(Xt, dtype=np.float64)
        else:
            cat_cols, num_cols = self._resolve_categorical_and_numeric_columns(X)
            self.categorical_columns_, self.numeric_columns_ = cat_cols, num_cols
            transformers = []
            if num_cols:
                if self.quantile_transform is None:
                    numerical_transformer = MinMaxScaler(clip=True)
                else:
                    numerical_transformer = self._make_quantile_transformer(n_samples=Xt.shape[0])
            
                transformers.append(("num", numerical_transformer, num_cols))
            if cat_cols:
                transformers.append(("cat", self._make_one_hot_encoder(), cat_cols))
            if not transformers:
                raise ValueError("No input columns available for preprocessing")
            self.preprocessor_ = ColumnTransformer(transformers, remainder="drop")
            Xt = np.ascontiguousarray(self.preprocessor_.fit_transform(X), dtype=np.float64)
            self.n_transformed_features_ = Xt.shape[1]
            self._build_internal_metadata()

        if not np.isfinite(Xt).all():
            raise ValueError("Transformed X contains NaN or infinite values.")
        
        tolerance = 1e-12
        
        if np.any(Xt < -tolerance) or np.any(Xt > 1.0 + tolerance):
            raise ValueError("Transformed X values must lie in [0, 1].")
        
        Xt = np.ascontiguousarray(np.clip(Xt, 0.0, 1.0), dtype=np.float64)

        start = time.perf_counter()
        encoded_candidates = self._sample_unique_candidates_encoded()
        self.sampling_time_ = time.perf_counter() - start
        features, states, modifiers, lengths, n_rules = encoded_candidates
        if features.shape[0] == 0:
            raise RuntimeError("Random search generated no valid candidates.")

        start = time.perf_counter()
        if self.threshold == 0.5:
            confusion_counts = _batch_confusion_threshold_half(
                    Xt,
                    y_binary,
                    features,
                    states,
                    modifiers,
                    lengths,
                    n_rules,
            )
        else:
            confusion_counts = _batch_confusion(
                Xt,
                y_binary,
                features,
                states,
                modifiers,
                lengths,
                n_rules,
                self.threshold,
            )

        objective_values = self._calculate_objective_values(*confusion_counts)
        self.evaluation_time_ = time.perf_counter() - start
        best_index = self._select_best_encoded(
            objective_values, modifiers, lengths, n_rules
        )
        self.rules_struct_ = self._decode_ruleset(
            features[best_index], states[best_index], modifiers[best_index],
            lengths[best_index], int(n_rules[best_index])
        )
        self.objective_value_ = float(objective_values[best_index])
        self.rule_stats_ = self._compute_rule_stats(Xt, y_binary, self.rules_struct_)
        self.rules_ = self._format_rules(self.rules_struct_, self.rule_stats_)
        self._compiled_rule_arrays_ = self._encode_single_ruleset(self.rules_struct_)
        return self

    def decision_function(self, X):
        """Calculate raw fuzzy RuleSet scores for input observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input observations. Input requirements follow those used during fitting.
        
        Returns
        -------
        scores : ndarray of float64, shape (n_samples,)
            Sum of selected rule activations for each observation.
        
        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.
        
        ValueError
            If the input does not satisfy scikit-learn validation requirements.
        
        Notes
        -----
        The returned score is non-negative. It is converted to a positive-class
        probability by ``predict_proba`` using
        
        ``p = 1 - exp(2 * log(0.5) * score)``.
        """
        check_is_fitted(self, ["rules_struct_", "_compiled_rule_arrays_"])
        self._check_X_no_return(X)
        if self.preprocessed:
            Xt = np.array(X, dtype=np.float64, order="C", copy=True)
        
            if Xt.shape[1] != self.n_features_in_:
                raise ValueError("X has a different number of features than the data passed to fit.")
        
            if np.any(Xt < 0.0) or np.any(Xt > 1.0):
                raise ValueError("Preprocessed X values must lie in [0, 1].")
        
            if self.quantile_transformer_ is not None:
                Xt[:, self.quantile_feature_indices_] = (
                    self.quantile_transformer_.transform(Xt[:, self.quantile_feature_indices_])
                )
        
        else:
            Xt = np.ascontiguousarray(self.preprocessor_.transform(X), dtype=np.float64)
        
        if not np.isfinite(Xt).all():
            raise ValueError("Transformed X contains NaN or infinite values.")
        
        tolerance = 1e-12
        
        if np.any(Xt < -tolerance) or np.any(Xt > 1.0 + tolerance):
            raise ValueError("Transformed X values must lie in [0, 1].")
        
        np.clip(Xt, 0.0, 1.0, out=Xt)
        Xt = np.ascontiguousarray(Xt, dtype=np.float64)
        return _score_rules_fast(Xt, *self._compiled_rule_arrays_)

    def predict_proba(self, X):
        """Estimate negative- and positive-class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input observations.
        
        Returns
        -------
        probabilities : ndarray of float64, shape (n_samples, 2)
            Class-probability matrix. The first column corresponds to
            ``negative_class_`` and the second column corresponds to
            ``positive_class_``.
        
        Notes
        -----
        For a raw RuleSet score ``s``, the positive probability is calculated as
        
        ``p_positive = 1 - exp(2 * log(0.5) * s)``.
        
        The negative probability is ``1 - p_positive``.
        """
        score = self.decision_function(X)
        p = 1.0 - np.exp(2.0 * np.log(0.5) * score)
        return np.column_stack((1.0 - p, p))

    def predict(self, X):
        """Predict binary class labels for input observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input observations.
        
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted labels expressed in the original target-label space.
        
        Notes
        -----
        An observation is assigned to ``positive_class_`` when the positive-class
        probability is greater than or equal to ``threshold``. Otherwise, it is
        assigned to ``negative_class_``.
        """
        positive = self.predict_proba(X)[:, 1] >= self.threshold
        return np.where(positive, self.positive_class_, self.negative_class_)

    def _make_quantile_transformer(self, n_samples):
        """Create the configured continuous-feature quantile transformer.

        Returns
        -------
        transformer : QuantileTransformer or Pipeline
            Transformation applied to continuous input features.
    
            If ``quantile_transform="uniform"``, the returned object is a
            ``QuantileTransformer`` mapping each feature independently to an
            approximately uniform marginal distribution on ``[0, 1]``.
    
            If ``quantile_transform="normal"``, the returned object is a pipeline
            containing a ``QuantileTransformer`` with a normal output distribution,
            followed by ``MinMaxScaler(clip=True)``. The final scaling is required
            because the fuzzy membership functions expect values in ``[0, 1]``.
    
        Notes
        -----
        The transformer is fitted only on continuous features from the training
        data. One-hot-encoded categorical columns are passed through unchanged.
    
        Quantile transformation is performed independently for each feature and
        reduces the influence of marginal outliers. The transformation is
        nonlinear and can alter linear relationships between input features.
    
        The normal-output variant is not used directly by the fuzzy membership
        functions. Its output is subsequently mapped to ``[0, 1]`` by a fitted
        min-max scaler.
        """
        if self.quantile_transform is None:
            return None
    
        n_quantiles = min(1000, n_samples) # 1000 - default value for QuantileTransformer
    
        quantile_transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=self.quantile_transform,
            subsample=10_000,
            random_state=self.random_state,
            copy=True,
        )
    
        if self.quantile_transform == "uniform":
            return quantile_transformer
    
        return Pipeline([
            ("quantile", quantile_transformer),
            ("unit_interval", MinMaxScaler(clip=True)),
        ])

    def _sample_unique_candidates_encoded(self):
        """Generate encoded valid unique candidate RuleSets.

        The method builds transformed-feature metadata required by the compiled
        generator, selects a random seed, calls ``_generate_unique_candidates_fast``,
        and records sampling diagnostics on the estimator.
        
        Returns
        -------
        features : ndarray of int64, shape (n_generated, max_rules, max_rule_length)
            Encoded feature indices of generated candidates.
        
        states : ndarray of uint8, shape (n_generated, max_rules, max_rule_length)
            Encoded condition-state codes.
        
        modifiers : ndarray of uint8, shape (n_generated, max_rules, max_rule_length)
            Encoded condition modifiers.
        
        lengths : ndarray of int64, shape (n_generated, max_rules)
            Active rule lengths.
        
        n_rules : ndarray of int64, shape (n_generated,)
            Active number of rules in every candidate.
        
        Warns
        -----
        UserWarning
            If fewer than ``n_candidates`` valid unique candidates are generated
            before reaching ``max_sampling_attempts``.
        
        Notes
        -----
        The method sets the following fitted diagnostic attributes:
        
        - ``n_sampling_attempts_``;
        - ``n_unique_candidates_``;
        - ``n_invalid_candidates_``;
        - ``n_duplicate_candidates_``;
        - ``valid_sampling_attempts_``;
        - ``duplicate_rate_``;
        - ``invalid_rate_``.
        """
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

    def _calculate_objective_values(
        self,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    ):
        """Calculate candidate objective values from confusion-matrix counts.

        Parameters
        ----------
        true_positives : ndarray of shape (n_candidates,)
            Number of true-positive predictions produced by each candidate RuleSet.
        
        true_negatives : ndarray of shape (n_candidates,)
            Number of true-negative predictions produced by each candidate RuleSet.
        
        false_positives : ndarray of shape (n_candidates,)
            Number of false-positive predictions produced by each candidate RuleSet.
        
        false_negatives : ndarray of shape (n_candidates,)
            Number of false-negative predictions produced by each candidate RuleSet.
        
        Returns
        -------
        objective_values : ndarray of float64, shape (n_candidates,)
            Objective-function value for every candidate. Larger values indicate
            better candidate performance.
        
        Raises
        ------
        ValueError
            If the custom ``objective_fun`` cannot be converted to a real scalar for
            any candidate.
        
        ValueError
            If the custom ``objective_fun`` returns a non-finite value for one or more
            candidates.
        
        Notes
        -----
        If ``objective_fun`` is ``None``, the Matthews correlation coefficient is
        calculated for all candidates by the dedicated Numba-compiled
        ``_batch_mcc_from_confusion`` function. Candidates for which the MCC
        denominator is zero receive an objective value of zero.
        
        If a custom objective function is supplied, it is called once per candidate
        with the signature
        
        ``objective_fun(tp, tn, fp, fn) -> float``.
        
        The custom function is executed outside Numba and must return one finite real
        scalar. Objective values are maximized during candidate selection.
        """
        if self.objective_fun is None:
            return _batch_mcc_from_confusion(
                true_positives,
                true_negatives,
                false_positives,
                false_negatives,
            )
    
        objective_values = np.empty(true_positives.shape[0], dtype=np.float64)
    
        for candidate_index in range(true_positives.shape[0]):
            value = self.objective_fun(
                true_positives[candidate_index],
                true_negatives[candidate_index],
                false_positives[candidate_index],
                false_negatives[candidate_index],
            )
    
            try:
                objective_values[candidate_index] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "objective_fun must return one real scalar "
                    "value for each candidate. "
                    f"Candidate {candidate_index} returned "
                    f"{value!r}."
                ) from error
    
        invalid_indices = np.flatnonzero(~np.isfinite(objective_values))
    
        if invalid_indices.size > 0:
            raise ValueError(
                "objective_fun returned non-finite values "
                "for candidate indices: "
                f"{invalid_indices[:10].tolist()}."
            )
    
        return objective_values

    @staticmethod
    def _select_best_encoded(objective_values, modifiers, lengths, n_rules):
        """Select the best encoded candidate by objective value and complexity.

        Candidates are compared first by objective value. Ties are resolved
        lexicographically using:
        
        1. total number of active conditions,
        2. number of active rules,
        3. sum of modifiers of active conditions.
        
        Parameters
        ----------
        objective_values : ndarray of float, shape (n_candidates,)
            Training objective value of every candidate.
        
        modifiers : ndarray of uint8, shape (n_candidates, max_rules, max_rule_length)
            Encoded condition modifiers.
        
        lengths : ndarray of int, shape (n_candidates, max_rules)
            Active lengths of candidate rules.
        
        n_rules : ndarray of int, shape (n_candidates,)
            Number of active rules in every candidate.
        
        Returns
        -------
        best : int
            Index of the selected candidate.
        
        Notes
        -----
        Only modifiers belonging to active conditions are included in the modifier
        sum. Padding modifiers do not affect model complexity.
        """
        best = 0
    
        best_modifier_sum = 0
    
        for rule_index in range(n_rules[0]):
            for literal_index in range(lengths[0, rule_index]):
                best_modifier_sum += int(modifiers[0, rule_index, literal_index])
    
        for candidate_index in range(1, len(objective_values)):
            better_objective = objective_values[candidate_index] > objective_values[best] + 1e-15
    
            same_objective = abs(objective_values[candidate_index] - objective_values[best]) <= 1e-15
    
            if better_objective:
                best = candidate_index
    
                best_modifier_sum = 0
    
                for rule_index in range(n_rules[best]):
                    for literal_index in range(lengths[best, rule_index]):
                        best_modifier_sum += int(modifiers[best, rule_index, literal_index])
    
            elif same_objective:
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
        """Decode numerical RuleSet arrays into immutable Condition objects.

        Parameters
        ----------
        features : ndarray of shape (max_rules, max_rule_length)
            Transformed-feature indices of one candidate.
        
        states : ndarray of uint8, shape (max_rules, max_rule_length)
            Encoded condition-state codes.
        
        modifiers : ndarray of uint8, shape (max_rules, max_rule_length)
            Encoded condition modifiers.
        
        lengths : ndarray of shape (max_rules,)
            Number of active conditions in every rule.
        
        n_rules : int
            Number of active rules to decode.
        
        Returns
        -------
        rules : tuple of tuple of Condition
            Immutable structured RuleSet representation.
        """
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
        """Convert an encoded condition-state integer to its string representation.

        Parameters
        ----------
        code : int
            Condition-state code:
        
            - 0: high,
            - 1: low,
            - 2: present,
            - 3: absent.
        
        Returns
        -------
        state : str
            Corresponding linguistic condition state.
        
        Raises
        ------
        IndexError
            If ``code`` is outside the supported interval [0, 3].
        """
        return ("high", "low", "present", "absent")[code]

    def _encode_single_ruleset(self, rules):
        """Encode one structured RuleSet into fixed-shape numerical arrays.

        Parameters
        ----------
        rules : tuple of tuple of Condition
            Structured RuleSet to encode.
        
        Returns
        -------
        features : ndarray of int64, shape (n_rules, max_rule_length)
            Transformed-feature indices. Padding positions contain ``-1``.
        
        states : ndarray of uint8, shape (n_rules, max_rule_length)
            Encoded condition-state codes.
        
        modifiers : ndarray of uint8, shape (n_rules, max_rule_length)
            Encoded condition modifiers. Padding positions contain one.
        
        lengths : ndarray of int64, shape (n_rules,)
            Number of active conditions in every rule.
        
        Notes
        -----
        The resulting arrays are stored after fitting and reused by
        ``decision_function``.
        """
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
        """Calculate support, lift, and coverage counts for selected rules.

        Rule coverage is determined by crisp thresholds derived from the fuzzy
        conditions:
        
        - High: ``x >= 0.5 ** (1 / modifier)``;
        - Low: ``x <= 1 - 0.5 ** (1 / modifier)``;
        - Present: ``x >= 0.5``;
        - Absent: ``x < 0.5``.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_transformed_features)
            Dense transformed training matrix with values in [0, 1].
        
        y : ndarray of bool, shape (n_samples,)
            Binary target where ``True`` denotes the positive class.
        
        rules : tuple of tuple of Condition
            Selected structured RuleSet.
        
        Returns
        -------
        stats : list of RuleStats
            One statistics object for every positive-class rule.
        
        Notes
        -----
        Support is the fraction of all observations covered by a rule.
        
        Lift is calculated as
        
        ``precision_among_covered / positive_class_prior``.
        
        A rule covering no observations receives zero lift.
        """
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
        """Format selected rules and their statistics as readable text.

        Parameters
        ----------
        rules : tuple of tuple of Condition
            Selected structured RuleSet.
        
        stats : sequence of RuleStats
            Rule statistics aligned with ``rules``.
        
        Returns
        -------
        output : list of str
            Human-readable positive-class rules followed by the default negative-class
            rule.
        
        Notes
        -----
        Each positive rule includes its lift, support, and number of covered training
        observations. Class names are taken from ``class_names`` when supplied;
        otherwise the strings ``"0"`` and ``"1"`` are used.
        """
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
        """Convert one structured condition to a human-readable antecedent.

        Parameters
        ----------
        condition : Condition
            Condition to format.
        
        Returns
        -------
        text : str
            Readable linguistic description of the condition.
        
        Notes
        -----
        Continuous high and low conditions use the prefixes ``Very`` and
        ``Extremely`` for modifiers two and three. Categorical conditions are
        formatted as category presence or absence using transformed-feature metadata.
        """
        meta = self.transformed_feature_metadata_[condition.feature]
        name = meta["name"]
        if condition.state in {"high", "low"}:
            prefix = {1: "", 2: "Very ", 3: "Extremely "}[condition.modifier]
            return f"{name} is {prefix}{condition.state.title()}"
        category = meta["category"]
        return f"{name} is {category}" if condition.state == "present" else f"{name} is not {category}"

    def _validate_parameters(self):
        """Validate estimator hyperparameters and resolve the rule-length limit.

        The effective maximum rule length is taken from ``max_rule_len`` when that
        parameter is not ``None``; otherwise ``max_rules_len`` is used.
        
        Raises
        ------
        ValueError
            If any size or sampling-budget parameter is smaller than one.
        
        ValueError
            If ``max_literal_repetitions`` is outside [1, 3].
        
        ValueError
            If ``sampling_type_number`` is not one or two.
        
        ValueError
            If ``sampling_type_length`` is not one or two.
        
        ValueError
            If ``threshold`` is outside [0, 1].
        
        ValueError
            If preprocessed mode is enabled without a continuous-feature
            specification.

        ValueError
            If ``quantile_transform`` is not ``None``, ``"uniform"``,
            or ``"normal"``.

        ValueError
            If ``objective_fun`` is neither callable nor ``None``.
        
        Notes
        -----
        The method sets ``_effective_max_rules_len_``.
        """
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

        if self.quantile_transform not in {None, "uniform", "normal"}:
            raise ValueError("quantile_transform must be None, 'uniform', or 'normal'.")

        if self.objective_fun is not None and not callable(self.objective_fun):
            raise ValueError("objective_fun must be callable or None.")

    def _configure_preprocessed_metadata(self, n_features, names):
        """Validate and store metadata for externally preprocessed input.

        Parameters
        ----------
        n_features : int
            Number of transformed columns in the supplied preprocessed matrix.
        
        names : sequence of str
            Names of transformed input columns.
        
        Raises
        ------
        ValueError
            If ``continuous_features`` contains unknown names or invalid indices.
        
        ValueError
            If categorical feature groups are required but not provided.
        
        ValueError
            If categorical groups overlap or contain invalid indices.
        
        ValueError
            If continuous and categorical feature specifications overlap.
        
        ValueError
            If any transformed feature is left unspecified.
        
        Notes
        -----
        The method sets:
        
        - ``continuous_feature_indices_``;
        - ``categorical_feature_groups_``;
        - ``transformed_feature_metadata_``;
        - ``numeric_columns_``;
        - ``categorical_columns_``.
        
        For categorical transformed columns, preprocessed mode does not recover the
        original category label automatically. The transformed column name is used
        for display metadata.
        """
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
        """Build transformed-feature metadata from the fitted preprocessor.

        The method reconstructs the exact transformed-column order emitted by the
        internal ``ColumnTransformer``. Numerical transformed features are listed
        first, followed by one-hot encoded categorical columns grouped by their
        original feature.
        
        Returns
        -------
        None
        
        Notes
        -----
        The method sets:
        
        - ``transformed_feature_metadata_``;
        - ``transformed_feature_names_``;
        - ``continuous_feature_indices_``;
        - ``categorical_feature_groups_``.
        
        For categorical features, category values are obtained from the fitted
        ``OneHotEncoder.categories_`` attribute.
        """
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
        """Resolve an index, name, or Boolean-mask feature specification.

        Parameters
        ----------
        spec : sequence of int, str, or bool
            Feature specification. Accepted forms are:
        
            - integer feature indices;
            - feature names;
            - a Boolean mask whose length equals ``n_features``.
        
        n_features : int
            Total number of available features.
        
        names : sequence of str
            Feature names used to resolve string specifications.
        
        parameter_name : str
            Name of the estimator parameter, used in validation messages.
        
        Returns
        -------
        result : set of int
            Resolved zero-based feature indices.
        
        Raises
        ------
        ValueError
            If string values do not match known feature names.
        
        ValueError
            If the specification mixes unsupported value types.
        
        ValueError
            If one or more resolved indices are outside the valid feature range.
        """
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
        """Determine names of original input features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        names : list of str
            Resolved input-feature names.
        
        Raises
        ------
        ValueError
            If the supplied ``feature_names`` length does not equal the number of
            input columns.
        
        Notes
        -----
        Feature names are resolved in the following order:
        
        1. explicitly supplied ``feature_names``;
        2. pandas DataFrame column names;
        3. automatically generated names ``x1``, ``x2``, and so on.
        """
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
        """Resolve original categorical and numerical input columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Original input data.
        
        Returns
        -------
        categorical : list
            Categorical columns expressed as indices or DataFrame column labels,
            depending on the input representation.
        
        numerical : list
            Numerical columns expressed as indices or DataFrame column labels,
            depending on the input representation.
        
        Raises
        ------
        ValueError
            If the explicit ``categorical_features`` specification is invalid.
        
        Notes
        -----
        If ``categorical_features`` is ``None`` and ``X`` is a pandas DataFrame,
        columns with ``object``, ``category``, or ``bool`` dtype are treated as
        categorical. All remaining columns are treated as numerical.
        
        For non-DataFrame input and no explicit categorical specification, every
        feature is treated as numerical.
        """
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
        """Create a dense one-hot encoder compatible with multiple sklearn versions.
        
        Returns
        -------
        encoder : OneHotEncoder
            Encoder configured to return dense arrays and ignore categories that
            were not observed during fitting.
        
        Notes
        -----
        Recent scikit-learn versions use ``sparse_output=False``. Older versions use
        ``sparse=False``. The method attempts the recent API first and falls back to
        the older parameter name.
        """
        try:
            return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover
            return OneHotEncoder(sparse=False, handle_unknown="ignore")

    def _column_to_index(self, column):
        """Convert a column identifier to its zero-based original-feature index.

        Parameters
        ----------
        column : int or str
            Integer column index or original feature name.
        
        Returns
        -------
        index : int
            Zero-based position in ``_input_feature_names_``.
        
        Raises
        ------
        ValueError
            If a string column name is not present in ``_input_feature_names_``.
        """
        return int(column) if isinstance(column, (int, np.integer)) else self._input_feature_names_.index(str(column))

    def _check_X_y_no_return(self, X, y):
        """Validate training features and target without returning converted arrays.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        
        y : array-like of shape (n_samples,)
            Training target.
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If ``X`` and ``y`` have inconsistent lengths, invalid dimensions, or
            contain non-finite values.
        
        Notes
        -----
        The method supports both recent and older scikit-learn validation APIs.
        Recent versions use ``ensure_all_finite`` while older versions use
        ``force_all_finite``.
        """
        try:
            check_X_y(X, y, dtype=None, ensure_all_finite=True)
        except TypeError:  # older sklearn
            check_X_y(X, y, dtype=None, force_all_finite=True)

    def _check_X_no_return(self, X):
        """Validate an input feature matrix without returning a converted array.

        The method delegates input validation to ``sklearn.utils.validation.check_array``
        and is intended for prediction-time validation, where only the feature matrix
        is available. The validated array returned by scikit-learn is deliberately
        discarded because subsequent preprocessing or conversion is handled by the
        calling method.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix to validate.
        
        Returns
        -------
        None
            The method performs validation only and does not return the validated or
            converted array.
        
        Raises
        ------
        ValueError
            If ``X`` is not a valid two-dimensional feature matrix, contains
            non-finite values, or otherwise violates the requirements imposed by
            ``sklearn.utils.validation.check_array``.
        
        Notes
        -----
        The method supports multiple scikit-learn versions. Recent versions use the
        ``ensure_all_finite`` keyword, whereas older versions use
        ``force_all_finite``. If the recent keyword is not supported, validation is
        repeated with the older keyword.
        
        Because the result returned by ``check_array`` is discarded, this method
        should not be used when the caller needs the converted NumPy array. Its
        purpose is limited to validating the original input before the estimator
        performs its own transformation or conversion.
        """
        try:
            check_array(X, dtype=None, ensure_all_finite=True)
        except TypeError:
            check_array(X, dtype=None, force_all_finite=True)
