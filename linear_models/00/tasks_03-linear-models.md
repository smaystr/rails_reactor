Homework 3
-----

You have [2 data sets](http://ps2.railsreactor.net/datasets/medicine/) and you need to use different linear models for each of them. You'll also need to answer some questions listed in the bottom of the task.
Task should be done as python scripts, 1 script per dataset. 
Answers should be provided in README.md file in the folder next to scripts.

<b>Notes:</b>

* Write your own implementation of Logistic and Linear Regression (with similar interface as [scikit-learn models](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))
* Compare your implementation by accuracy/precision/mse and scikit-learn 

You can use:

* [numpy](https://www.numpy.org/) for calculations
* [argparse](https://docs.python.org/3/library/argparse.html) for params parsing
* [pathlib](https://docs.python.org/3/library/pathlib.html) for handling the dataset folder structure



### Data set 1. Heart Disease UCI

This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient (0 or 1). 

#### Attribute Information: 
1. age 
2. sex 
3. chest pain type (4 values) 
4. resting blood pressure 
5. serum cholestoral in mg/dl 
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results (values 0,1,2)
8. maximum heart rate achieved 
9. exercise induced angina 
10. oldpeak = ST depression induced by exercise relative to rest 
11. the slope of the peak exercise ST segment 
12. number of major vessels (0-3) colored by flourosopy 
13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

You need to predict `target`.
There is a folder with `heart_train.csv` and `heart_test.csv` files. First you use for training and validation, second - use only for final solution and check results.

### Data set 2. Medical Cost Personal Datasets

#### Attribute Information: 

- age: age of primary beneficiary
- sex: insurance contractor gender, female, male
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
- children: Number of children covered by health insurance / Number of dependents
- smoker: Smoking
- region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
- charges: Individual medical costs billed by health insurance

You need to predict `charges`.
There is a folder with `insurance_train.csv` and `insurance_test.csv` files. First you use for training and validation, second - use only for final solution and check results.


### Please answer following questions:
1. For which tasks is it better to use Logistic Regression instead of other models?
2. What are the most important parameters that you can tune in Linear Regression / Logistic Regression / SVM?
3. How does parameter C influence regularisation in Logistic Regression?

4. Which top 3 features are the most important for each data sets?
5. Which accuracy metrics did you receive on train and test sets for `Heart Disease UCI` dataset?
6. Which MSE did you receive on train and test datasets for `Medical Cost Personal`? 
6. *(additional)** Try a few different models and compare them. <i>(You can use scikit-learn models for this)</i>


### Deadline - next Thursday 04.07.2019
---
### Please ask all questions regarding homework in Discord
