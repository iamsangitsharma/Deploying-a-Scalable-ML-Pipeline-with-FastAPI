# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created by Sangit Sharma as part of the Udacity Machine Learning DevOps course. It is a Random Forest classifier from scikit-learn (version 1.5.1). The model uses 100 trees, a maximum tree depth of 15 and a random state of 42 so the results can be reproduced. Categorical features are one-hot encoded and the label is binarized using the `process_data` function in `ml/data.py`.

## Intended Use
The model predicts whether a person earns more than $50K a year based on census information such as age, education, occupation and hours worked per week. It is intended for learning purposes and to show how a model can be trained, tested and served through a REST API with FastAPI. It should not be used to make real decisions about individual people, such as hiring, lending or insurance.

## Training Data
The data is the Census Income dataset (also known as the "Adult" dataset) from the UCI Machine Learning Repository, extracted from the 1994 US Census database. It contains 32,561 rows and 15 columns: 6 numeric features, 8 categorical features and the label `salary`. The classes are imbalanced: about 76% of people earn <=50K and about 24% earn >50K. 80% of the data (26,048 rows) was used for training. The split was stratified on `salary` so both sets keep the same class balance.

## Evaluation Data
The remaining 20% of the data (6,513 rows) was held out as the test set. It was processed with the same OneHotEncoder and LabelBinarizer that were fitted on the training data, so no information from the test set was used during training.

## Metrics
The model was evaluated with precision, recall and F1 score, where the positive class is >50K.

| Metric | Score on test set |
|---|---|
| Precision | 0.7932 |
| Recall | 0.5918 |
| F1 | 0.6779 |

This means when the model predicts >50K it is correct about 79% of the time, but it finds only about 59% of the people who really earn >50K.

Performance was also measured on slices of the data for every value of every categorical feature, and the full results are in `slice_output.txt`. For example, the F1 score is 0.6806 for males and 0.6617 for females. By race, F1 ranges from 0.4615 for Amer-Indian-Eskimo (73 people) to 0.6818 for White (5,533 people).

## Ethical Considerations
The dataset includes sensitive attributes such as race, sex and native country, and the model uses them as features. The slice results show that performance is not equal across groups, especially for small groups like Amer-Indian-Eskimo where recall is only 0.3333. The data is from 1994, so it can reflect historical bias in pay between groups. Using this model for real decisions could unfairly affect people from these groups.

## Caveats and Recommendations
The data is old and income levels have changed since 1994, so the model should not be used on current data without retraining. Some slices contain very few people (for example, one person in "Never-worked"), so their metrics are not reliable. Recall is lower than precision, so the model misses many high earners; tuning the classification threshold or using class weights could improve this. A fairness tool such as Aequitas could be used to study bias in more detail, and the model should be retrained and re-checked on slices whenever new data is used.
