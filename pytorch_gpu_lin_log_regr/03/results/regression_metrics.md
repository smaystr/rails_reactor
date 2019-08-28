**Program started with command**:
```
main.py --script 2 --task 2 --config config_regression.json --on_gpu True
```
**Task**: *regression*

**Original values**:
```
tensor([ 9095.0684,  5272.1758, 29330.9824,  9301.8936, 33750.2930,  4536.2588,
         2117.3389, 14210.5361,  3732.6250, 10264.4424], device='cuda:0')
```
**Predicted values**:
```
tensor([ 9553.1572,  7729.3989, 39277.1914,  9702.1602, 28834.8633, 11014.8262,
          650.2432, 17214.9102,  1348.7949, 11529.0449], device='cuda:0',
       grad_fn=<SliceBackward>)
```
**Scores**:
- **MSE (Mean Squared Error)**: *33701988.0*
- **RMSE (Root Mean Squared Error)**: *5805.341333634053*
- **MAE (Mean Absolute Error)**: *4162.90234375*
- **R-Squared (Coefficient of Determination)**: *0.7829161842580155*
- **MPE (Mean Percentage Error)**: *-26.337924599647522*
- **MSPE (Mean Square Percentage Error)**: *45.50565481185913*
- **MAPE (Mean Absolute Percentage Error)**: *47.08269536495209*
