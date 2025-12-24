'''
This script is used to run the the pipeline of MemGuard. 
'''
import os 
import configparser

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

# Ensure result folder is set relative to repo
result_folder = os.path.normpath(os.path.join(script_dir, '..', 'result', 'location', 'code_publish'))
os.makedirs(result_folder, exist_ok=True)

if 'location' not in config:
    raise KeyError('location')
config["location"]["result_folder"] = result_folder + '/'

with open(config_path, 'w') as configfile:
    config.write(configfile)

cmd = f"python {os.path.join(script_dir, 'train_user_classification_model.py')} -dataset location"
os.system(cmd)

cmd = f"python {os.path.join(script_dir, 'train_defense_model_defensemodel.py')} -dataset location"
os.system(cmd)

cmd = f"python {os.path.join(script_dir, 'defense_framework.py')} -dataset location -qt evaluation"
os.system(cmd)

cmd = f"python {os.path.join(script_dir, 'train_attack_shadow_model.py')} -dataset location -adv adv1"
os.system(cmd)

cmd = f"python {os.path.join(script_dir, 'evaluate_nn_attack.py')} -dataset location -scenario full -version v0"
os.system(cmd)
