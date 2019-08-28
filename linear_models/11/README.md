**For which tasks is it better to use Logistic Regression instead of other models?**<br>
 For modeling a binary dependent variable, for target data where  probabilities for modeling are close to 0 or 1

**What are the most important parameters that you can tune in Linear Regression / Logistic Regression / SVM?**<br>
Linear Regression
* max iteration
* learning rate

Logistic Regression
* C
* solver
* max iteration
* learning rate

SVM
* kernel
* gamma
* C

**How does parameter C influence regularisation in Logistic Regression?**<br>
Small values of C increase the regularization strength which creates simple models that underfit the data. 
Big values of C low the power of regularization and may lead to overfit.

**Which top 3 features are the most important for each data sets?**<br>
Heart Disease UCI
* cp
* thalach
* slope

Medical Cost Personal Datasets
* smoker 
* age
* bmi

**Which accuracy metrics did you receive on train and test sets for Heart Disease UCI dataset?**<br>
train 0.8471074380165289<br>
test 0.8032786885245902

**Which MSE did you receive on train and test datasets for Medical Cost Personal?**

train 37277681.70201866 <br>
test 33596915.85136151

