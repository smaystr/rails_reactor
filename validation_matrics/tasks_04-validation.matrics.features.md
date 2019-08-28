Homework 4
-----
You need to implement utility for end2end model training lifecycle using model from your previous hw.
Utility accepts:
* **[required]** dataset (path to local .csv file or url)
* **[required]** target variable name
* **[required]** task: classification/regression
* **[required]** path for output model: `output.info` `output.model`
* **[optional with defaults]** parameter for validation split type: train-test split/k-fold/leave one-out
* **[optional with defaults]** parameter for validation split size
* **[optional with defaults]** parameter for specifying time series column to perform timeseries validation 
* **[optional with defaults]** parameter for hyperparameter fitting algo: grid search/random search (also add parameter for this)

Use files(datasets) from previous hw to test you script.
You can fill records that have N/A values with mean or drop column that have both numerical and categorical data. (Consider adding 'column_value_exists' boolean column for such features)

The `output.info` file should contain:
* all metrics that your can calculate for specific task
* info about training phase(time, loss)
* info about feature importance

The `output.model` file should contain:
* type (classificator/regressor)
* best hyperparameters
* weights


<b>Notes:</b>

You can use:

* [numpy](https://www.numpy.org/) for calculations
* [argparse](https://docs.python.org/3/library/argparse.html) for params parsing
* [pathlib](https://docs.python.org/3/library/pathlib.html) for handling the dataset folder structure

Feel free to ask about package you want to use... We just want to hold off using high level ds stuff like `pandas`, `sklearn`, etc..


Deadline - next Thursday 09.07.2019
---
Please ask all questions regarding homework in Discord
