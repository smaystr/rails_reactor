# PyTorch Linear Models report
Running model on GPU is better, but not so much, because we don't have 
a lot of data like in NN. 

If we have `bigger learning rate` or/and `smaller batch size`, it will
converges much faster.

In the `charts` folder you can check models' metrics results

if you wanna test it by yourself, please, install `tensorboard` with
`pip` and run from this directory after `heart_logit.py` or
`insurance_linreg.py` execution 
```
tensorboard --logdir="./logs" 
```
