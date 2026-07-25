# Spatial Architecture Classifier Baseline

## Objective

The objective was to evaluate whether sample-level spatial and
cell-composition features can classify the tumor–immune architecture
of MIBI-TNBC samples.

The prediction target was `patient_class`:

| Code | Label |
|---:|---|
| 0 | mixed |
| 1 | compartmentalized |
| 2 | cold |

This is a portfolio and methodological experiment. It is not a
clinically validated prediction model.

## Dataset and evaluation design

The modeling table contained one row per sample:

- 40 total samples
- 19 mixed
- 15 compartmentalized
- 6 cold

The existing sample-level train, validation, and test assignments
were retained.

For model selection:

- the train and validation samples were combined into a
  32-sample development set;
- the 8 test samples remained untouched;
- model comparison used repeated stratified five-fold
  cross-validation with three repetitions;
- the primary selection metric was macro F1.

Because each sample appears exactly once in the sample-level
feature table, no sample could occur in both the training and
validation portion of the same fold.

## Feature sets

Three feature sets were compared.

### Count-only

The count-only feature set contained:

- total cell count;
- tumor-cell count;
- CD8 T-cell count;
- macrophage count;
- B-cell count;
- corresponding cell fractions.

### Distance-only

The distance-only feature set contained sample-level summaries of:

- CD8 T-cell distance to the nearest tumor cell;
- macrophage distance to the nearest tumor cell;
- B-cell distance to the nearest tumor cell;
- indicators for samples in which a relevant immune population
  was absent.

Distances were represented in approximate micrometers.

### Combined

The combined feature set contained both count and distance features.

Pixel-distance columns were excluded because they are exact scaled
duplicates of the micrometer-distance columns.

## Models

The following models were compared:

- dummy most-frequent classifier;
- elastic-net logistic regression;
- random forest;
- histogram gradient boosting.

Median imputation was performed inside each model pipeline.
Logistic-regression inputs were standardized. Tree models were
constrained to reduce overfitting on the small dataset.

## Cross-validation results

| Model | Feature set | Mean macro F1 | Standard deviation | Mean balanced accuracy |
|---|---|---:|---:|---:|
| HistGradientBoosting | combined | 0.804 | 0.189 | 0.819 |
| Random forest | combined | 0.802 | 0.181 | 0.830 |
| Elastic-net logistic regression | count-only | 0.760 | 0.157 | 0.800 |
| Random forest | count-only | 0.740 | 0.171 | 0.759 |
| HistGradientBoosting | count-only | 0.706 | 0.215 | 0.719 |
| Random forest | distance-only | 0.647 | 0.206 | 0.685 |
| Dummy most frequent | not applicable | 0.213 | 0.011 | 0.333 |

Histogram gradient boosting with the combined feature set was
selected using mean macro F1.

The difference between histogram gradient boosting and random
forest was very small relative to the variation across folds.
The results therefore do not establish that one tree model is
inherently superior.

## Held-out test results

The selected model was evaluated once on the eight held-out samples.

| Metric | Value |
|---|---:|
| Accuracy | 0.625 |
| Balanced accuracy | 0.722 |
| Macro F1 | 0.714 |
| Weighted F1 | 0.625 |

Five of eight test samples were classified correctly.

### Class-level performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Mixed | 0.667 | 0.500 | 0.571 | 4 |
| Compartmentalized | 0.500 | 0.667 | 0.571 | 3 |
| Cold | 1.000 | 1.000 | 1.000 | 1 |

The single cold test sample was classified correctly. This does
not establish reliable performance for the cold class because its
test support was only one sample.

## Error analysis

Three samples were misclassified:

| SampleID | True label | Predicted label |
|---:|---|---|
| 13 | mixed | compartmentalized |
| 29 | mixed | compartmentalized |
| 36 | compartmentalized | mixed |

All errors occurred between mixed and compartmentalized tumors.

This is biologically plausible because tumor–immune mixing exists
along a continuum rather than as a perfectly sharp boundary.
The current feature set may not capture all local contacts and
boundary structures required to separate borderline samples.

## Count versus spatial information

Combined features improved both nonlinear models relative to their
count-only versions:

| Model | Count-only macro F1 | Combined macro F1 | Difference |
|---|---:|---:|---:|
| HistGradientBoosting | 0.706 | 0.804 | +0.098 |
| Random forest | 0.740 | 0.802 | +0.062 |

Distance-only models were weaker than count-only or combined
models.

These results provide preliminary evidence that spatial distance
features add information beyond basic cell abundance, but that
distance summaries are most useful when interpreted together with
cell composition.

## Permutation importance

Cross-validated permutation importance identified two leading
candidate features:

| Feature | Family | Mean importance | Standard deviation | Positive fraction |
|---|---|---:|---:|---:|
| `tumor_fraction` | cell fraction | 0.291 | 0.211 | 0.663 |
| `cd8_to_tumor_mean_um` | distance | 0.144 | 0.148 | 0.677 |
| `macrophage_to_tumor_mean_um` | distance | 0.002 | 0.085 | 0.063 |

Tumor fraction was the strongest composition feature.

Mean CD8-to-tumor distance was the strongest spatial feature,
supporting the hypothesis that immune–tumor separation contributes
to architecture classification.

Importance estimates were unstable and should be treated as
exploratory. Many distance summaries are highly correlated, so
permutation importance may assign little or no importance to one
feature when similar information remains available in another.

## Limitations

1. The dataset contains only 40 modeled samples.
2. The cold class contains only six samples.
3. The held-out test set contains only eight samples.
4. Cross-validation results vary substantially across folds.
5. Several distance summaries are strongly correlated.
6. Distances use an approximate conversion of 0.39 micrometers
   per pixel.
7. The current spatial table includes nearest-neighbor distances
   but not the complete planned contact, density, neighborhood,
   or graph feature sets.
8. The target represents published tumor–immune architecture
   labels, not treatment response or clinical outcome.
9. The results are exploratory and not clinically validated.

## Conclusion

A sample-level spatial architecture classifier was successfully
implemented using leakage-safe evaluation.

Cell composition was the strongest source of predictive signal.
Mean CD8-to-tumor distance contributed additional spatial
information, and combined features outperformed count-only
features for both nonlinear classifiers.

The main classification difficulty was separating mixed from
compartmentalized tumors. Additional contact, density, boundary,
and neighborhood features may improve this distinction.