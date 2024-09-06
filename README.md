# **Image Classification using our own neural network model and Transfer Learning Repository**

This repository contains various **experiments** and **scripts** used to train and fine-tune models. It includes comparisons of different architectures, optimizers, and training strategies. Each folder contains saved model, plots with loss, acurracy and performance comparison results. 

Detailed **presentation of our project**, including plots and descriptions.

[presentation in english](https://www.canva.com/design/DAGP7cwss4Q/iwJalHHFrlgvEFytPdbOUw/view?utm_content=DAGP7cwss4Q&utm_campaign=designshare&utm_medium=link&utm_source=editor)

[presentation in polish](https://www.canva.com/design/DAGCnfvyrGs/Cd4GIMKMWnXS3HWC1JVJyA/view?utm_content=DAGCnfvyrGs&utm_campaign=designshare&utm_medium=link&utm_source=editor)


### **Python Script Files**

- **`check_parameters.py`**  
  Used to **validate and print model parameters** to ensure the correct setup before training begins.

- **`eval.py`**  
  Script to **evaluate model performance** on validation or test datasets after training is complete.

- **`train_epoch.py`**  
  Defines the **training loop** for each epoch, used across different models and configurations.
  
- **`script.py`**  
**MOST IMPORTANT FILE**  
This file is used for the entire training process and integrates other scripts such as `eval.py` and `train_epoch.py`. You can easily modify the model, number of epochs, or optimizer, and then simply run the script. After the training is complete, a folder is generated that contains the saved model and corresponding plots. The plots show the loss and accuracy during the training process, allowing us to monitor whether the model is overfitting or learning too slowly.


## **Key Experiments**

1. **Optimizer Comparison**  
   Compare the performance of models trained with different optimizers:  
   - `big_model_Adam`  
   - `big_model_RMS`  
   - `big_model_SGD`

2. **Effect of Dropout**  
   Evaluate the impact of **dropout layers** on the model's overfitting using `big_model_with_dropout`.

3. **EfficientNet with and without Transforms**  
   Compare the results of model trained with and without data augmentation:  
   - `enet_no_transforms`  
   - `enet_with_transforms`

4. **ResNet Fine-Tuning and Linear Probing**  
   - `resnet_fine_tuning`: Full fine-tuning of a ResNet model.  
   - `resnet_linear_probing`: Training only the final layers while freezing the rest of the model.

5. **Small vs Large Model**  
   Assess how a **smaller model** (`small_model`) compares with larger models in terms of accuracy and training efficiency.

