### **Task**: *classification*

#### **Model**: *Logistic Regression*

# Script 1

**Hyperparams**:
```json
{
    "C": 100,
    "learning_rate": 0.01,
    "num_iterations": 2
}
```

**Device**: 
- *cpu*:
   - **fitting time**: *0.043881893157958984s*
- *cuda* (Nvidia GTX 1060 6Gb):
   - **fitting time**: *0.5505383014678955s*

**Scores**:
- **Accuracy**: *0.8360655737704918*
- **Precision**: *0.9259259259259259*
- **Recall (Sensitivity)**: *0.7575757575757576*
- **Specificity**: *0.9285714285714286*
- **F1 Metric**: *0.8333333333333334*
- **Roc-Auc Metric**: *0.8430735930735931*
- **Log-loss**: *2.2644059658050537*

# Script 2

**Hyperparams**:
```json
{
    "learning_rate": 0.01,
    "epochs": 5
}
```

**Device**: 
- *cpu*:
   - **fitting time**: *0.35208916664123535s*
- *cuda* (Nvidia GTX 1060 6Gb):
   - **fitting time**: *1.5933306217193604s*

**Scores**:
- **Accuracy**: *0.9016393442622951*
- **Precision**: *0.8857142857142857*
- **Recall (Sensitivity)**: *0.9393939393939394*
- **Specificity**: *0.8571428571428571*
- **F1 Metric**: *0.9117647058823529*
- **Roc-Auc Metric**: *0.8982683982683983*
- **Log-loss**: *1.3580381870269775*
