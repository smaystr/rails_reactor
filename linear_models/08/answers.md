# Homework #3 answers
1. Logistic Regression suits perfectly for classification tasks. In some cases, SVM will be better, but it we may come into many pitfalls with this model.
2. If we talk about hyper parameters, we can tune the following:
- learning rate
- regularization
- class weights(must tune for unbalanced data)
- number of iterations
- solver(liblinear/saga/newton-cg...) .
3. C is inverse regularization parameter, so the bigger C value, the less regularization your model will get and will be less overfitting-resistant.
4. For the heart disease dataset following features was the most important:
- feature 23
- feature 2
- feature 21
and for the medical insurance:
- feature 5 (OHE feature)
- feature 0 (age)
- feature 1 (sex)
5. Metrics on Heart Disease UCI:
- ACCURACY:0.85246
- PRECISION:0.85294
- RECALL:0.87879
- F1:0.86567
6. Metrics on Medical Cost Personal:
- MSE: 0.00888
- RMSE: 0.09429
- MAE: 0.06802

Inversed metrics:
- MSE:33597819.18255
- RMSE:5796.36258
- MAE:4181.41727

7.Basic [sklearn logit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from the box scored 0.843 on accuracy metric.

