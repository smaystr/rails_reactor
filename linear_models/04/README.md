1. Logistic Regression is used for classification. It's better to use this model when **y** takes only
two values (binary). Also, it used when we think that explanatory **X**-variables are related to the 
**y** variable.

2. The most important parameters are learning rate and number of iterations. 
    - **Learning rate** determines how fast or slow we will move towards the optimal weights. 
    If the it is very large we will skip the optimal solution. 
    If it is too small we will need too many iterations to converge to the best values.
    - Wrong choice of **number of iterations** may cause overfitting, or underfitting problems.

3. Parameter **C** (*inverse regularisation parameter*) determines the strength of the regularization.
**Higher** values of C correspond to **less** regularization.

4. **Heart Disease UCI** dataset:
    - chest pain type;
    - maximum heart rate achieved;
    - the slope of the peak exercise ST segment;
    - resting electrocardiographic  results.
    
    **Medical Cost Personal** dataset:
    - smoker;
    - age;
    - charges;
    - children.

5. **Heart Disease UCI** dataset.
    ```
    - MSE (Mean Squared Error): 41187292
    - RMSE (Root Mean Squared Error): 6418
    - MAE (Mean Absolute Error): 4218
    - R-Squared (Coefficient of Determination): 0.73
    - MSPE (Mean Square Percentage Error): 52
    - MAPE (Mean Absolute Percentage Error): 45
    ```
    **scikit-learn model**:
    ```
    - MSE (Mean Squared Error): 33596916
    - RMSE (Root Mean Squared Error): 5796
    - MAE (Mean Absolute Error): 4181
    - R-Squared (Coefficient of Determination): 0.78
    - MSPE (Mean Square Percentage Error): 47
    - MAPE (Mean Absolute Percentage Error): 47
    ```
6. **Medical Cost Personal** dataset.

    **Warning** best results on my model achieved with regularization param = 980.
    I don't know why.
    ```
    - Confusion matrix:
        - TP: 28
        - FP: 5
        - FN: 5
        - TN: 23 
    - Accuracy: 0.84
    - Precision: 0.85
    - Recall: 0.85
    - F1 Metric: 0.85
    ```
    **scikit-learn model**:
    ```    
    - Accuracy: 0.89
    - Precision: 0.86
    - Recall: 0.94
    - F1 Metric: 0.90
    ```
