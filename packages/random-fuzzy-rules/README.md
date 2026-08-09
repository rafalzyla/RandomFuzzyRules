# Random Fuzzy Rules

`random-fuzzy-rules` provides a scikit-learn-compatible implementation of the
`RandomFuzzyRulesClassifier`. The classifier learns compact and interpretable
fuzzy rule sets by randomly generating candidate model structures, rejecting
invalid or duplicate candidates, and selecting the candidate with the best
training objective.

Candidate evaluation is accelerated with Numba-compiled computational kernels.
The implementation is designed for current scientific Python environments and
integrates with standard scikit-learn workflows.

## Main features

- scikit-learn-compatible `fit`, `predict`, and `predict_proba` interface;
- support for continuous features and one-hot-encoded categorical feature
  groups;
- explicit limits on the number of rules, rule length, and literal repetitions;
- configurable sampling distributions for rule-set size and rule length;
- reproducible randomized search through the `random_state` parameter;
- compact fuzzy rule-set representation intended for model inspection;
- support for dense, preprocessed input matrices with values in `[0, 1]`;
- Numba-accelerated candidate evaluation.

## Basic usage

```python
from random_fuzzy_rules import RandomFuzzyRulesClassifier

classifier = RandomFuzzyRulesClassifier(
    max_rules=6,
    max_rules_len=3,
    max_literal_repetitions=3,
    threshold=0.5,
    n_candidates=10_000,
    max_sampling_attempts=20_000,
    sampling_type_number=1,
    sampling_type_length=2,
    preprocessed=True,
    continuous_features="all",
    random_state=42,
)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
```

When `preprocessed=True`, the input matrix must be dense, finite, and scaled to
the interval `[0, 1]`. Categorical variables should be represented by
one-hot-encoded column groups and identified through the
`categorical_feature_groups` parameter.

## Model interpretation

The fitted classifier exposes the selected fuzzy rule set for inspection. Each
rule is a conjunction of fuzzy conditions, while the complete model aggregates
the activations of its rules to obtain the classification score.

A fitted rule set can be displayed with:

```python
print(classifier.rules_)
```

Individual rules can be printed separately:

```python
for rule_index, rule in enumerate(classifier.rules_, start=1):
    print(f"Rule {rule_index}: {rule}")
```

The limits on the number of rules, rule length, and literal repetitions provide
explicit control over the trade-off between predictive performance and model
complexity.

## Configuration used in the experiments

The default configuration selected through the accompanying ablation study is:

```python
{
    "max_rules": 6,
    "max_rules_len": 3,
    "max_literal_repetitions": 3,
    "threshold": 0.5,
    "n_candidates": 10_000,
    "max_sampling_attempts": 20_000,
    "sampling_type_number": 1,
    "sampling_type_length": 2,
}
```

The complete ablation, scalability, and classifier-comparison workflows are
available in the repository-level `experiments` directory. Cross-platform
reproduction scripts are provided in the `scripts` directory.

## License

This package is distributed under the MIT License. See the repository-level
`LICENSE` file for the complete license text.
