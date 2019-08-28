# Text preprocessing
- Lemmatization to cast all words to the bases, therefore decrease number of words in the corpus
- stopwords removal for russian and ukrainian to get rid of useless words 

## Metrics MSE/MAE, inference for Decision Tree and CatBoost with Td-Idf and SVD features 
Decision Tree:
- MSE train - 283,563,771     MSE test - 1,226,182,844
- MAE train - 9,367      	  MAE test - 14,043

CatBoost 
- MSE train - 90,526,313   	  MSE test - 1,575,715,928
- MAE train - 5,785 		  MAE test - 16,629

## Metrics MSE/MAE, inference for Neural Network with FastText

 * 2 hidden layers with 512 units; activation: relu
 - MSE train - 2,124,737,300  MSE test - 2,424,059,400
 - MAE train - 21,469		  MAE test - 22,257

 * 3 hidden layers with 512 units; activation: relu
 - MSE train - 1,783,459,600  MSE test - 2,060,405,200
 - MAE train - 22,731 		  MAE test - 23,592

 * 4 hidden layers with 512 units; activation: relu
 - MSE train - 2,249,108,700  MSE test - 2,541,933,800
 - MAE train - 22,261		  MAE test - 23,029
