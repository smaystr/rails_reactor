# Homework 6

**Goal:** explore PyTorch and test training models on GPU.

# Tasks

1. Implement linear and logistic regression in PyTorch:
    - Essentially convert the code you developed for HW3 from NumPy to PyTorch
    - As a first step, keep it as close as possible to the NumPy implementation: PyTorch API supports a lot of things available in NumPy (i.e., linear algebra operations)
2. Also, port your gradient descent implementation to PyTorch:
    - If you use all of the data points (or only a single one) in your implementation, implement mini-batch [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD)
    - Make sure the SGD parameters are configurable, either via a configuration file (preferred; you can use JSON, YAML, etc.) or via a command-line argument
    - Use the datasets from HW3 to evaluate your PyTorch implementation
3. Test your implementations on CPU:
    - Use various evaluation metrics to catch any errors
    - Compare the performance of NumPy and PyTorch implementations
4. Train your models on GPU (*script 1*):
    - Check PyTorch documentation to find out how you can run your code using CUDA
    - Add a command-line argument that would control whether the code runs on CPU or GPU
    - If you do not have access to a GPU, use [Google Colab](https://colab.research.google.com/) or [Kaggle kernels](https://www.kaggle.com/kernels)
    - Make sure you select a GPU runtime, and that your code is actually running on a GPU
    - Hand in the Python script in the merge request, not a link to some online notebook
    - Evaluate the speed up, *report* results
    - Try increasing or decreasing the batch size in your SGD, *report* the impact on performance, speed of convergence, and metrics (accuracy, etc.). It is a good idea to plot the training process (loss vs. time) for different values of batch size and learning rate.
    - (Optional) You can use [TensorBoard](https://www.tensorflow.org/tensorboard/) ([PyTorch documentation](https://pytorch.org/docs/stable/tensorboard.html)) to track and visualize the training metrics. This is optional since it was not covered in the lectures, but it is valuable to familiarize yourselves with this tool.
5. Convert your implementation to idiomatic PyTorch code (*script 2*):
    - Subclass `torch.nn.Module` for your implementation of linear / logistic regression, use `torch.nn.Linear`, loss, etc. available in PyTorch API
    - Try to use PyTorch’s `DataLoader` for loading datasets
    - Evaluate the results and performance with PyTorch’s SGD, and try out other optimizers (such as Adam), *report* the results (speed of convergence and metrics)

## Script parameters
Your script should accept the same parameters as in HW3, as well as:
- A parameter that controls whether the code should be run on CPU or GPU
- Model / training parameters if you decide to pass them via command-line arguments. It is better to use the config file for this though, in which case you would need to pass the location of this file.

# What to hand in

The things that you should include in your merge request are mentioned in the 'Tasks' section in *italics*:

- **Script 1.** PyTorch implementation of linear and logistic regression. Should be a simple port of NumPy code to PyTorch. Models should be trained using mini-batch SGD. The script should accept the argument that controls whether it is run on CPU or GPU.
- **Script 2.** An idiomatic implementation of your linear and logistic regression models in PyTorch that make extensive use of PyTorch APIs. Again, it should be runnable on both CPU and GPU.
- A **markdown file** (or two) with the report on the requested results, metrics and plots.

# Deadline

**Due on 03.08.2019 23:59**
