1. Logistic regression is a great baseline for classification tasks and when the probability of a class is needed rather than a prediction. They are simple and easy to interpret.
2. Most important hyperparameters to tune in LogReg and LinReg are learning rate, regularization rate and regularization penalty. For a lower learning rate tolerance might need to be tuned.
3. Hyperparameter 'C' is the inverse regularization term. Meaning that default value of 1 does no regularization and usually is in the range of [0,1], the lower - the heavier is the regularization penalty.
4. Most important features for heart dataset are: cp, sex, thal
   Most important features for cost dataset are: 
5. Train score: 0.8553719008264463
   Test  score: 0.8032786885245902
6. Train MSE: 37277681.7030143
   Test  MSE: 33596908.84305915
7. Random forest beats both algorithms in regression and classification.