GPU and CPU:
The model trains faster on the GPU than CPU. However, if the data is very small, it may be the opposite.

Batch size:
With increasing of batch size model trains longer, however loss decreases and vice versa.

Learning rate:
With increasing of learning rate model converges faster however but if lr is too large, then it can diverge. With decreasing of learning rate model converges slower but if lr is too small it may not reach the minimum.

SGD and Adam:
Adam really works better than a gradient descent. Time to reach the minimum decreases by 0.1-0.4 seconds (depending on dataset size).

Fixed lr and pytorch scheduler (ReduceLROnPlateau):
Using ReduceLROnPlateau also accelerates the algorithm in addition to the previous improvement. Time decreases approximately the same (0.1-0.4 s). Given that the datasets are small and the time to converge before was also small, then this can be considered a significant acceleration.
