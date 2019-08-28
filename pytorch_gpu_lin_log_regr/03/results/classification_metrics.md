**Program started with command**:
```
main.py --script 2 --task 1 --config config_classification.json --on_gpu True
```
**Task**: *classification*

**Original values**:
```
tensor([0., 0., 0., 0., 0., 0., 1., 0., 1., 0.], device='cuda:0')
```
**Predicted values**:
```
tensor([0., 0., 0., 1., 0., 0., 1., 0., 1., 0.], device='cuda:0',
       grad_fn=<SliceBackward>)
```
**Scores**:
- **Accuracy**: *0.9016393442622951*
- **Precision**: *0.8857142857142857*
- **Recall (Sensitivity)**: *0.9393939393939394*
- **Specificity**: *0.8571428571428571*
- **F1 Metric**: *0.9117647058823529*
- **Roc-Auc Metric**: *0.8982683982683983*
- **Log-loss**: *1.3580384254455566*
