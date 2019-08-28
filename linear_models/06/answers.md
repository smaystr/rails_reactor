1. Logistic regression is efficient, doesn't require many computational resources. It is quite simple so it cat be implemented relatively easy and quick. However you can use it when you have several classes but it is not that efficient when you have multi-class classification problem.

2. The most important parameter in linear and logistic regression is learning rate alpha. It is very important to find the correct alpha value. If it is too large, the gradient descent will slip through tha minimum and it will not converge (or can even diverge), if it is too small - then it will take a lot of iterations to reach the minimum and converge. Also a very important parameter is the regularization parameter. If the parameter is too large, the coefficients will turn out to be very small which will lead to underfitting. And vice versa: if C is small, then such a regularization will not help prevent overfitting.

3. Just as it was described in the previous answer: if the parameter is too large, the coefficients will turn out to be very small which will lead to underfitting; and vice versa: if C is small, then such a regularization will not help prevent overfitting.

4. Logistic Regression: chest pain type, maximum heart rate achieved, the slope of the peak exercise ST segment, resting electrocardoigraphic results. Linear Regression: smoker, age, bmi, children

5. Train: 0.85, test: 0.81

6. Train: 37277681, test: 33612787