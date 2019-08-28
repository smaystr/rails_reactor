# REPORT 1: LOW LEVEL MODELS

*My implementation of GD with numpy without test train split:*

* model score for test data is: *0.8688524590163934*

* model score for train data is: *0.871900826446281*

* model fit time is *0.59 s*

_SK learn_ model score for test data is: *0.8688524590163934*

*My implementation of SGD with pytorch:*

* model fit time is *5.75 s* on _CPU_

* CPU model score for test data is: *0.8852459016393442*

* CPU model score for train data is: *0.871900826446281*

* model fit time is *13.48 s* on _Kaggle GPU_

* GPU model score for test data is: *0.8852459016393442*

* GPU model score for train data is: *0.871900826446281*

Model metrics (params: lr = 0.01, num_iter = 10000, C = 0.1, batch_size = 32):
              
              accuracy: 0.8852459016393442
              recall: 0.90625
              precision: 0.8787878787878788
              f1: 0.8923076923076922
              log-loss: 0.3219337463378906

Max score: *0.9016393442622951*, fit time: *0.81 s*, params: {'C': 0.01, 'num_iterations': 2000, 'learning_rate': 0.01, 'batch_size': 32}
 
               accuracy: 0.9016393442622951
               recall: 0.90625
               precision: 0.90625
               f1: 0.90625
               log-loss: 0.31819549202919006
                
Min fit time: *0.26 s*, params: {'C': 0.01, 'num_iterations': 1000, 'learning_rate': 0.5, 'batch_size': 64}
 
               accuracy: 0.8688524590163934
               recall: 0.875
               precision: 0.875
               f1: 0.875
               log-loss: 0.3901352882385254

_Also I can load full report for the params tuning with time dependency_

# REPORT 2: HIGH LEVEL PYTORCH MODELS

### CPU: 

Model params: batch size - 32, optimizer - Adam, criterion - BCELoss, epochs - 1000

Model metrics: 

               accuracy: 0.8524590134620667
               recall: 0.8125
               precision: 0.8965517282485962
               f1: 0.8524590134620667
               log-loss: 0.16357722878456116

Fit time:

          4.376444101333618 s

### Kaggle GPU:

Model params: batch size - 32, optimizer - Adam, criterion - BCELoss, epochs - 1000

Model metrics: 

               accuracy: 0.8524590134620667
               recall: 0.8125
               precision: 0.8965517282485962
               f1: 0.8524590134620667
               log-loss: 0.1602255403995514

Fit time:

          8.610266208648682 s
      
