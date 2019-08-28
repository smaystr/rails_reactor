1. Model trains faster on GPU. In some cases like small dataset it can on the opposite
2. Batch size affects learning time and gradient estimate. Will small batch size model trains faster, but decrease accuracy of your gradient stimates. And on the opposite, when batch size is big enough you increase gradient estimation accuracy and decrease learning speed.
3. Small learning rate converges model long enough, and big learning rate on the reverse converges model fast. But with big learning rate the model can deiverge.
