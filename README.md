# **Model Training and Comparison Repository**

This repository contains various **experiments** and **scripts** used to train and fine-tune models. It includes comparisons of different architectures, optimizers, and training strategies. Each folder contains model-specific configurations, training scripts, and performance comparison results.

## **Repository Structure**

### **Model Plots Folders**

- **`best_model/`**  
  *Best model trained for 50 epochs using the Adam optimizer.*

- **`big_model/`**  
  *Big model trained with the Adam optimizer for 50 epochs, used in model comparison experiments.*

- **`big_model_Adam/`, `big_model_RMS/`, `big_model_SGD/`**  
  *These folders contain models trained using different optimizers (Adam, RMSprop, SGD) for 50 epochs, comparing their performance.*

- **`big_model_with_dropout/`**  
  *Model with dropout layers added, trained with Adam optimizer for 50 epochs to test overfitting and generalization performance.*

- **`enet_no_transforms/`** and **`enet_with_transforms/`**  
  *Model trained with and without data augmentations/transforms to compare their effect.*

- **`resnet_fine_tuning/`**  
  *ResNet model fine-tuned for 50 epochs using the AdamW optimizer.*

- **`resnet_linear_probing/`**  
  *ResNet model trained using linear probing for 50 epochs with AdamW optimizer. Only final layers are trained while most of the network remains frozen.*

- **`small_model/`**  
  *A smaller model trained for 50 epochs using the Adam optimizer, used to compare performance with larger models.*

### **Python Model Files**

- **`small_model.py`**  
  Contains the architecture for the **small model**.

- **`big_model.py`**  
  Model definition for the **large model** trained with the Adam optimizer.

- **`big_model_with_dropout.py`**  
  Same as `big_model.py` but with **dropout layers** to reduce overfitting.
    
- **`best_model.py`**  
  Contains the architecture for the **best performing model**.

- **`enet.py`**  
  Script containing architecture for example model.
  
- **`resnet_fine_tuning.py`**  
  Fine-tunes ResNet for 50 epochs with the AdamW optimizer, focusing on fine-tuning pre-trained weights.

- **`resnet_linear_probing.py`**  
  Implements **linear probing** for ResNet, training only the final layers of the model.

### **Python Script Files**

- **`check_parameters.py`**  
  Used to **validate and print model parameters** to ensure the correct setup before training begins.

- **`eval.py`**  
  Script to **evaluate model performance** on validation or test datasets after training is complete.

- **`script.py`**  
  General utility script that likely contains shared **helper functions** or logic used in multiple training experiments.

- **`train_epoch.py`**  
  Defines the **training loop** for each epoch, used across different models and configurations.

## **Key Experiments**

1. **Optimizer Comparison**  
   Compare the performance of models trained with different optimizers:  
   - `big_model_Adam`  
   - `big_model_RMS`  
   - `big_model_SGD`

2. **Effect of Dropout**  
   Evaluate the impact of **dropout layers** on the model's generalization performance using `big_model_with_dropout`.

3. **EfficientNet with and without Transforms**  
   Compare the results of model trained with and without data augmentation:  
   - `enet_no_transforms`  
   - `enet_with_transforms`

4. **ResNet Fine-Tuning and Linear Probing**  
   - `resnet_fine_tuning`: Full fine-tuning of a ResNet model.  
   - `resnet_linear_probing`: Training only the final layers while freezing the rest of the model.

5. **Small vs Large Model**  
   Assess how a **smaller model** (`small_model`) compares with larger models in terms of accuracy and training efficiency.

