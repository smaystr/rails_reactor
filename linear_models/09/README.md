+ _For which tasks is it better to use Logistic Regression instead of other models?_

Logistic Regression is used for binary classification. It is simple and highly interpretable.
It is good for obtaining some baseline solution which could be improved by more complex models.
Also data should be linearly separable in order to get good result with LR.

+ _What are the most important parameters that you can tune in Linear Regression / Logistic Regression / SVM?_
    
    + using normalization or not
    + regularization parameters
    + sigmoid shape for Logistic Regression (not sure if it possible with sklearn)
    + stop criteria (via some tolerance or iterations number)
    + initial coef values also may influence convergence time but not sure about solution quality
    + kernel type and params for SVM

+ _How does parameter C influence regularisation in Logistic Regression?_
    
    According to sklearn docs C is inverse of regularization param.
    I suppose it is something like a denominator for penalty in loss function.

+ _Which top 3 features are the most important for each data sets_?
    
    + insurance dataset:
        + smoker
        + children
        + bmi
    + heart dataset:
        For heart dataset I omitted categorical vars except sex and got a little bit contoversial results:
        + my own logistic regression implementation
            + thalach
            + negative oldpeak
            + negative trestpbs and sex
            
        + sklearn logistic with default params
            + negative oldpeak
            + sex
            + age, trestbps, chol, thalach - influence of all those params are negligible
    

+ _Which accuracy metrics did you receive on train and test sets for Heart Disease UCI dataset_?
    
    F1 score:
    + own implementation
        + train 72%
        + test 74%
    + sklearn
        + train 76%
        + test 73%

+ Which MSE did you receive on train and test datasets for Medical Cost Personal?
    + own implementation
        + train 37369745
        + test 33990122
    + sklearn
        + train 37369582
        + test 33979257

+ (additional)* Try a few different models and compare them. (You can use scikit-learn models for this)
