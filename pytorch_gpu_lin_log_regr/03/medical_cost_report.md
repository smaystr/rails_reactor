### **Task**: *regression*

#### **Model**: *Linear Regression*

# Script 1

**Hyperparams**:
```json
{
    "C": 1,
    "learning_rate": 0.1,
    "num_iterations": 2
}
```

**Device**: 
- *cpu*:
   - **fitting time**: *0.0359039306640625s*
- *cuda* (Nvidia GTX 1060 6Gb):
   - **fitting time**: *0.7370278835296631s*

**Scores**:
- **MSE (Mean Squared Error)**: *33292884.0*
- **RMSE (Root Mean Squared Error)**: *5769.998613518031*
- **MAE (Mean Absolute Error)**: *4113.6689453125*
- **R-Squared (Coefficient of Determination)**: *0.7855513361474324*
- **MPE (Mean Percentage Error)**: *-23.498499393463135*
- **MSPE (Mean Square Percentage Error)**: *43.250927329063416*
- **MAPE (Mean Absolute Percentage Error)**: *45.20939290523529*

# Script 2

**Hyperparams**:
```json
{
    "learning_rate": 0.005,
    "epochs": 2
}
```

**Device**: 
- *cpu*:
   - **fitting time**: *0.5016579627990723s*
- *cuda* (Nvidia GTX 1060 6Gb):
   - **fitting time**: *2.258958339691162s*

**Scores**:
- **MSE (Mean Squared Error)**: *33701988.0*
- **RMSE (Root Mean Squared Error)**: *5805.341333634053*
- **MAE (Mean Absolute Error)**: *4162.90234375*
- **R-Squared (Coefficient of Determination)**: *0.7829161842580155*
- **MPE (Mean Percentage Error)**: *-26.337924599647522*
- **MSPE (Mean Square Percentage Error)**: *45.50565481185913*
- **MAPE (Mean Absolute Percentage Error)**: *47.08269536495209*
