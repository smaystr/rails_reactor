GPU / CPU. My model trains a little bit faster on the GPU rather than  on CPU. 

Batch size. Increasing batch size helps model converge better (loss decrease) but needs more time for training.

Learning rate. As learning rate increases, the model converges faster, but if learning rate is too large, it can diverge. With a decrease in learning rate, the model converges more slowly and more reliably, but if learning rate is too small, it may not reach a minimum.

SGD / Adam. The Adam optimizer approaches a local minimum much faster than SGD. At the same time, metrics show almost the same as the SGD algorithm.


