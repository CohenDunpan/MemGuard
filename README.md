This repository contains code for "MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples". 

Requirements (tested on Python 3.9): PyTorch 2.5.1 (CUDA 12.1), torchvision 0.20.1, torchaudio 2.5.1, numpy, argparse, scipy. The project no longer depends on TensorFlow/Keras. GPU is optional but recommended.



# Dataset description: 

data/location contains the location dataset from the paper: "Membership Inference Attacks Against Machine Learning Models". You may consider citing this paper if you use this dataset. 

# Code usage: 


input_data_class.py is used to input the data needed for all other modules.

config.ini maintains the configuration for the experiments such as the hyperparameters. 

model folder contains the file that defines the machine learning models which include target model, attack model, and defense model. 

train_user_classification_model.py is used to train target model. 

train_defense_model_defensemodel.py is used to train defense model. 

defense_framework.py implements our defense algorithm. 

train_attack_shadow_model.py is used to train attacker's shadow model.

evaluate_nn_attack.py is used to evaluate the NN-Attack. 

All trained models are saved as PyTorch state_dict checkpoints in the result folder (e.g., epoch_XXX_weights_*.pt).


You can directly run run_location_defense.py (python run_location_defense.py) after installing python tools. It will automatically run the pipeline. 

We also run the code and obtain the following result: (similar to Figure 1 in MemGuard paper). 

Distortion budget: [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]

Inference accuracy: [0.50, 0.52, 0.55, 0.62, 0.69, 0.73]

This code was written by Jinyuan Jia. If you have any question, please feel free to send email to jinyuanjia02@gmail.com. 

# Citation
If you use this code, please cite the following paper: 
# <a href="https://arxiv.org/pdf/1909.10594.pdf">MemGuard</a>
```
@inproceedings{jia2019memguard,
  title={{MemGuard}: Defending against Black-Box Membership Inference Attacks via Adversarial Examples},
  author={Jinyuan Jia and Ahmed Salem and Michael Backes and Yang Zhang and Neil Zhenqiang Gong},
  booktitle={CCS},
  year={2019}
}
```