# Preference learning: Car Evaluation

## Dataset

**Car Evaluation** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation). The task is to predict **car acceptability** from six categorical attributes.


## Task and scale

- **Alternatives:** 1728 (all combinations of the six attributes in the original design).
- **Criteria / features:** 6 (ordinal, after encoding).
- **Target:** **4 classes** — multi-class classification; labels are not binarized for this project.

## Criteria (features)

After preprocessing used in this repository, every feature is numeric in `[0, 1]` with **higher values meaning more preferable** along that criterion: lower buying and maintenance cost, more doors and passenger capacity, larger luggage boot, and higher safety.

| Column | Name       | Description        | Levels (natural order, worst → best) | Encoding in CSV                          |
| ------ | ---------- | ------------------ | ------------------------------------ | ---------------------------------------- |
| 0      | `buying`   | Buying price       | vhigh, high, med, low                | 4 equally spaced: `0`, `1/3`, `2/3`, `1` |
| 1      | `maint`    | Maintenance price  | vhigh, high, med, low                | same as `buying`                         |
| 2      | `doors`    | Number of doors    | 2, 3, 4, 5 & more                    | same as `buying`                         |
| 3      | `persons`  | Passenger capacity | 2, 4, more                           | 3 equally spaced: `0`, `0.5`, `1`        |
| 4      | `lug_boot` | Luggage boot size  | small, med, big                      | same as `persons`                        |
| 5      | `safety`   | Estimated safety   | low, med, high                       | same as `persons`                        |

## Class labels (target)

Column 6 is the integer class id (multi-class). Mapping to the UCI string labels:

| Value in CSV | UCI label | Meaning (acceptability) |
| ------------ | --------- | ----------------------- |
| 1            | `unacc`   | Unacceptable            |
| 2            | `acc`     | Acceptable              |
| 3            | `good`    | Good                    |
| 4            | `vgood`   | Very good               |

Higher class id corresponds to higher acceptability.

## Experiments

[XGboost](notebooks/xgboost.ipynb)
[ANN-UTADIS](notebooks/ann_utadis.ipynb)
[Neural Network](notebooks/neural_network.ipynb)