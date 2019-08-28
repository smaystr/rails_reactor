1. Logistic Regression is used for classification i.e. tasks, where the target belongs to one of a finite amount of classes e.g. 1 or 0 for binary classification. It can be efficiently used in production because of its simple implementation (using simple mathematic operations, meaning it's easy to transfer to code) and the ability to get really good accuracy with a correct transformation of data. 
2. For a simple linear/logistic regression the main hyperparameters are learning rate and number of iterations for the optimizing algorithm, and also the regularization parameters, if you're using regularization. Can't say much about SVM as we haven't talked about it yet.  
3. C is the inverse of regularization strength, so the lesser the value, the stronger the regularization.
4. If you look at the correlation between target and features, it makes sense that the most important features are  
* thal, cp, exang (in that order) for the heart disease dataset
* smoker, age, bmi for the insurance dataset
5. For the Heart Disease UCI Dataset
* Train accuracy: 0.7355371900826446
* Test accuracy: 0.7213114754098361
* Train precision: 0.8207547169811321
* Test precision: 0.7857142857142857
6. For the Medical Cost Personal Dataset
* Train MSE: 128475007.01537958
* Test MSE: 131201335.64669806