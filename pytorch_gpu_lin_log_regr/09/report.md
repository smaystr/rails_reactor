<b>Numpy vs Torch performance:</b><br>
With the same results, numpy finishes 3 times faster than torch. <br>
<br>
<i>Numpy:</i><br>
    <li> 0.006334 seconds </li>
<i>Torch:</i><br>
    <li> 0.022298 seconds </li>
<br>
<b>CPU vs GPU torch speed:</b> 
<br>
Google colab was used as a service with gpu notebooks.<br>
<br>
<i>CPU:</i><br>
    <li> 0.2782 seconds</li><br>
<i>GPU:</i><br>
    <li> 0.3216 seconds</li><br>
<br>
<br>
<b>Performance profiling of different learning rates and batch sizes</b><br>
Evaluation dataset used was insurance_train.csv. Tolerance was picked judging by the best convergence time.<br>
<br>
<b> Batch size 16, learning rate 0.003 </b><br>
![16_003](images/learning_curve16_0.003.png)<br>
<i>Elapsed time:  0.12643694877624512<br>
Mean squared error : 37281232.0<br>
Mean absolute error : 4192.0986328125<br>
Explained Variance Score : 0.7417232990264893</i><br>
<br>
<b> Batch size 32, learning rate 0.003 </b><br>
![16_003](images/learning_curve32_0.003.png)<br>
<i>Elapsed time:  0.1248929500579834<br>
Mean squared error : 37283660.0<br>
Mean absolute error : 4203.05126953125<br>
Explained Variance Score : 0.7416884899139404</i><br>
<br>
<b> Batch size 48, learning rate 0.003 </b><br>
![16_003](images/learning_curve48_0.003.png)<br>
<i>Elapsed time:  0.10982012748718262<br>
Mean squared error : 37282084.0<br>
Mean absolute error : 4206.359375<br>
Explained Variance Score : 0.7416977882385254</i><br>
<br>
<b> Batch size 64, learning rate 0.003 </b><br>
![16_003](images/learning_curve64_0.003.png)<br>
<i>Elapsed time:  0.12246203422546387<br>
Mean squared error : 37278440.0<br>
Mean absolute error : 4203.87548828125<br>
Explained Variance Score : 0.741723895072937</i><br>
<br>
<br>
<br>
<b> Batch size 16, learning rate 0.1 </b><br>
![16_003](images/learning_curve16_0.1.png)<br>
<i>Elapsed time:  0.010492086410522461<br>
Mean squared error : 37298644.0<br>
Mean absolute error : 4214.109375<br>
Explained Variance Score : 0.7415899038314819</i><br>
<br>
<b> Batch size 32, learning rate 0.1 </b><br>
![16_003](images/learning_curve32_0.1.png)<br>
<i>Elapsed time:  0.013717889785766602<br>
Mean squared error : 37734364.0<br>
Mean absolute error : 4234.58154296875<br>
Explained Variance Score : 0.7388425469398499</i><br>
<br>
<b> Batch size 48, learning rate 0.1 </b><br>
![16_003](images/learning_curve48_0.1.png)<br>
<i>Elapsed time:  0.01411890983581543<br>
Mean squared error : 38075688.0<br>
Mean absolute error : 4250.453125<br>
Explained Variance Score : 0.7368855476379395</i><br>
<br>
<b> Batch size 64, learning rate 0.1 </b><br>
![16_003](images/learning_curve64_0.1.png)<br>
<i>Elapsed time:  0.015953779220581055<br>
Mean squared error : 37775532.0<br>
Mean absolute error : 4217.1513671875<br>
Explained Variance Score : 0.7387539148330688</i><br>
<br>
<br>
<b> PyTorch results on same dataset </b>

<b> Batch size 64, learning rate 0.7, epochs 25, ADAM </b><br>
![16_003](images/learning_curve_torch64_0.7.png)<br>
<i>Elapsed time:  14.604433059692383<br>
Mean squared error : 43920040<br>
Mean absolute error : 4172.35302734375<br>
Explained Variance Score : 0.7299135327339172</i><br>
<br>
<b> Batch size 256, learning rate 0.03, epochs 100, SGD, early_stopping </b><br>
![16_003](images/learning_curve_torch_SGD256_0.03.png)<br>
<i>Elapsed time:  0.8477602005004883<br>
Mean squared error : 38302136.0<br>
Mean absolute error : 4190.84840241255<br>
Explained Variance Score : 0.7348201853423612</i><br>
<br>