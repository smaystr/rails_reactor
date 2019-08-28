1. Running on GPU obviosly makes algorythm faster, but on small data it doesn't show it's max.
 For instance, on my laptop with Nvidia GeForce 940MX total train was even slightly worse

2. Increasing batch size helps model to converge faster but pays with speed

3. Learning rate is also can be choosen uncorrectly which leads to passing the best local minimum.
  However 0.001 works fine for me as usual. Bigger lr converges model faster, but it may pass optimal
  minimum

4. Adam and Nesterov works a little bit better than SGD

5. With tensorboard you can check metrics through training