## Cell phenotype classifier results

We trained supervised machine-learning models to predict broad MIBI cell groups from marker-expression profiles only. The task was designed as a tabular ML baseline and pipeline validation step, not as independent biological discovery.

The dummy classifier performed poorly, confirming that class imbalance alone cannot solve the task. Logistic regression performed well, but tree-based models performed substantially better. The best model was HistGradientBoosting, reaching accuracy = 0.992 and macro F1 = 0.972.

The remaining errors were biologically plausible:

1. The largest remaining error was between keratin-positive tumor cells and immune cells. This may reflect ambiguous cells, local marker noise, segmentation artifacts, or cells with weak canonical tumor/immune marker signal.

2. Immune cells were occasionally predicted as keratin-positive tumor cells. This is important because the MIBI dataset contains multiplex marker expression, and classification depends on combinations of markers rather than one perfect marker.

3. Logistic regression confused more immune cells with mesenchymal-like and keratin-positive tumor cells than the tree-based models. This suggests that simple linear boundaries are insufficient for some marker combinations.

4. Random forest and HistGradientBoosting greatly reduced the confusion errors, meaning non-linear models captured marker interactions better than logistic regression.

5. The remaining errors are concentrated in biologically adjacent or technically ambiguous groups, not random classes. This supports that the model learned real marker-expression structure, while also showing that broad cell-group labels are not perfectly separable.

Overall, this experiment validates the cleaned MIBI table, the leakage-safe split, the marker-expression features, and the supervised ML evaluation pipeline. Because the labels are marker-derived, the results should be interpreted as a baseline and quality-control step rather than a new biological finding.

## Model comparison

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Dummy most frequent | 0.502 | 0.167 | 0.111 | 0.335 |
| Logistic regression | 0.930 | 0.956 | 0.790 | 0.936 |
| Random forest | 0.975 | 0.964 | 0.954 | 0.975 |
| HistGradientBoosting | 0.992 | 0.978 | 0.972 | 0.992 |