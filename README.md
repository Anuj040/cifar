# cifar

## Installation Guideline

```
git clone git@github.com:Anuj040/cifar.git [-b <branch_name>]
cd cifar (Work Directory)

# local environment settings
pyenv local 3.6.9
python -m pip install poetry
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

# In case older version of pip throws installation errors
poetry run python -m pip install --upgrade pip 

# local environment preparation
poetry install

```

## Working with the code
All the code executions shall happen from work directory.

### Training mode
There are three different training modes for the classifier training, plus one for training of auto-encoder (AE) only

1. For auto-encoder only training, please execute the following 
```
poetry run python cifar/start.py --train_mode=pretrain --epochs=400
```
2. For classifier's training
  * classifier only training
  ```
  poetry run python cifar/start.py --train_mode=classifier --epochs=400
  ```
  * AE only training followed by classifier only training
  ```
  poetry run python cifar/start.py --train_mode=both --epochs=400
  ```
  * simultaneous AE-classifier training in _multitask setting_
  ```
  poetry run python cifar/start.py --train_mode=combined --epochs=400
  ```
### Saving the model files
1. For AE only or "both" training, at the end of AE training, a model file will be saved in the following directory.
```
    cifar {work_directory}
    ├── cifar
    ├── ae_model                   
    │   ├──ae_model.h5         # Saved model
    └── ...
```
2. For classifier only or "both" training, during classifier training, progressively better models will be saved in the following directory.
```
    cifar {work_directory}
    ├── cifar
    ├── class_model                   
    │   ├──class_model_{epoch:04d}_{val_acc:.4f}.h5         # Saved model # epoch & val_acc = epoch at which model was saved and corresponding validation accuracy
    └── ...
```
3. For "combined" training, during the training, progressively better models will be saved in the following directory.
```
    cifar {work_directory}
    ├── cifar
    ├── com_model                   
    │   ├──com_model_{epoch:04d}_{val_acc:.4f}.h5         # Saved model # epoch & val_acc = epoch at which model was saved and corresponding validation accuracy
    └── ...
```
**Note: By default the current implementation, splits the original training set of cifar-10 into train/val (80:20) split for classifier training.

### Evaluation 
To run the evaluation on test dataset, please execute the following from the work_directory
```
poetry run python cifar/start.py --train_mode=combined --model_path=com_model/com_model_0500_0.0000.h5 --mode=eval
```

### Inference
To run inference on a single image _eg: test0.png_, please execute the following from the work_directory
```
poetry run python cifar/start.py --train_mode=combined --model_path=com_model/com_model_0500_0.0000.h5 --mode=infer --img_path=./test0.png
```
** Note: please make sure to use the right combination of _train_mode_ and _model_path_ flags, _eg: as shown above or (classifier, class_model/class_model_0500_0.0000.h5)_.
