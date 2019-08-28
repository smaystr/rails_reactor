<b>REPORTING</b>
<br>
<b>Neural Network </b>
<br>
3 layers, 2048, relu, RMSE: train - 733859.75; validation - 744406.75<br>
3 layers, 512, relu, RMSE: train - 718444.8125; validation - 730227.5<br>
5 layers, 512, relu, RMSE: train - 720594.8125; validation - 732838.4375<br>
2 layers, 512, relu, RMSE: train - 724379.75; validation - 735113.0<br>
5 layers, 256, relu, RMSE: train - 719952.0625; validation - 732686.8125<br>
5 layers, 526, selu, RMSE: train - 778732.0239; validation - 785375.18<br>
HardTanh did not converge past 1m rmse with 5 layers, 512 units<br>
<br>
<b>Decision tree (great at overfitting)</b><br>
Train MAE: 161.25382595309244; Test MAE: 333926.20029875066<br>
<br>
<b>Ensemble of 4 LightGBM models with Bayesian Optimization</b><br>
Train MAE: 29220.809256736768; Test MAE: 207671.46329659596<br>
<br>
<b> Parameters </b><br>
CountVectorizer was used to extract text features. <br>
LabelEncoder and OneHot were used to encode columns containing strings.<br>
<br>
<b>std</b>
uah_price             718742.233807<br>
street                   942.782301<br>
region                     3.396685<br>
total_area                20.228990<br>
room_count                 0.892162<br>
construction_year        576.900047<br>
verified_price             0.415708<br>
verified_apartment         0.415708<br>
latitude                   1.815312<br>
longitude                  1.806436<br>
city                       4.870676<br>
<b>mean</b><br>
uah_price                    1349123.6112130478<br>
street                         1610.50499490316<br>
region                         8.61495073054706<br>
total_area                    57.79938837920489<br>
room_count                   1.9153924566768603<br>
construction_year             180.8387359836901<br>
latitude                      48.08775419813322<br>
longitude                    30.438677126415215<br>
city                         14.621542643560991<br>
<br>
<b>Validation parameters</b><br>
<b>Everything has a random_state 30, including models</b><br>
KFold 4 splits, shuffle<br>
train_test_split 0.25 test size, shuffle<br>
Bayesian Optimization 15 exploitation rounds, 5 exploration rounds
