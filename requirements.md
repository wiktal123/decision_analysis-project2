# Project Preference Learning

## 1 Introduction

In this project, you’ll conduct a comparative analysis using multiple methods
that solve the sorting problem. You’ll experiment with various approaches and
assess their performance on a chosen dataset. Find a data set that contains
a few monotonic criteria, over 100 alternatives, and at least 2 classes (if the
problem has more classes, you can binearize them).

- The entire project must be done individually or in pairs..
- The report can be made in a jupyter notebook (.ipynb + HTML) or as
  a python project + report in PDF.

Briefly describe the data set including the criteria descriptions. For this
dataset, train the following models:

1. One interpretable ML model (XGBoost or rankSVM or Logistic Regression)
2. One interpretable neural MCDA method (ANN-UTADIS)
3. Neural network with a few layers with nonlinear activation functions

## 2 Experiments

For each model:

- report Accuracy, F1 and AUC
- Models should be presented with all visualizations to facilitate interpretation.
- All presented values should be rounded to a maximum of 4 decimal places.
- A brief summary taking into account the conclusions of the following analyses

### 2.1 Explanation of the decisions

- Explain why the decision was made for the 3 selected alternatives.
- Take 3 alternatives and say what the minimum change to a single criterion
  should be done so that the option is classified into a different class.
  1. Try to answer this question in an analytical way based only on the
     values of the model parameters and explain why such a change is
     minimal (without sampling).
  2. Perform space sampling by slightly changing the evaluations of
     alternatives to get a different class. Do the results agree with theoretical
     predictions?

- Explain the predictions for these objects using at least one technique
  (SHAP, Individual Conditional Expectation)

### 2.2 Interpretation of the model

- Based on the parameters obtained, can we say something about the user’s
  preferences?
- What was the influence of the criteria? Are there any criteria that have
  no effect, or have a decisive influence?
- Are there any dependencies between the criteria?
- What is the nature of the criterion, gain, cost, non-monotonic?
- Whether there are any preference thresholds? Are there any evaluations
  on criteria that are indifferent in terms of preferences?
- Interpret the model by at least one (Partial Dependence Plot, Permutation
  Feature Importance, …)

A list of tools that contain various techniques for explaining and interpreting
the model:

- Shapash
- Alibi
- Explainerdashboard
- DALEX
- eli5

## 3 Grading

- 3 – One interpretable ML model with whole experiments
- 4 – Requirements for 3 + One interpretable neural MCDA method (ANN-UTADIS)
- 5 – Requirements for 4 + Neural network with a few layers

Note: ANN-UTADIS is a fully interpretable model, so explaining that you
can’t give some information because they are neural network is not enough.
