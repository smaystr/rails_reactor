# Answer for Questions

1. Logistic Regression works well for classification tasks. SVM can be more robust, but that highly depends on task.

2. In all linear models we can tune next parameters:
    - regularization type and regularization constant;
    - number of iterations;
    - class weights are the most valuable parameters for all tasks with unbalanced data.

3. The bigger C parameter, the less regularization is applied.

4. For the "heart_*.csv" dataset the most valuable are (all columns are parts of one-hot transformation):
    - FEATURE exang_0 with importance 1.0742432626774667;
    - FEATURE cp_0 with importance -1.0472944347125752;
    - FEATURE exang_2 with importance 0.9829213299178376;
    - FEATURE exang_3 with importance -0.7734570985920353;
    - FEATURE cp_2 with importance 0.7424127265153394.

   For the "insurance_*.csv" the most important are:
    - FEATURE smoker with importance 0.3845898653753915;
    - FEATURE age with importance 0.05880382338070916;
    - FEATURE bmi with importance 0.03312264740302293;
    - FEATURE region_0 with importance 0.03147604016629842;
    - FEATURE region_1 with importance 0.02543459319211285.

5. Metrics for Heart Disease UCI:

    ```md
    - ACCURACY:  0.83607
    - PRECISION: 0.81081
    - RECALL:    0.90909
    - F1-SCORE:  0.85714
    ```

6. Metrics for Medical Cost Personal:

    ```md
    - TRANSFORMED
    - MSE:  0.00889
    - RMSE: 0.09430
    - MAE:  0.06803

    - INVERSE TRANSFORMED
    - MSE:  33600423.08139
    - RMSE: 5796.58719
    - MAE:  4182.13610
    ```
