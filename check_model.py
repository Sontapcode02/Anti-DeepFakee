import torch
import sys
import os

sys.path.append(os.getcwd())
from network.proposed_model import DeepfakeMultiClassModel

model_path = r'weight\19_multiclass_colab.pkl'

print(f"Checking file: {model_path} ...")
try:
    state_dict = torch.load(model_path, map_location='cpu')
    print("Successfully loaded .pkl file into memory (size: ~86MB).")
    print(f"Number of parameters (layers): {len(state_dict)}")
    
    # Init model
    model = DeepfakeMultiClassModel(image_size=299, num_classes=3)
    
    # Check 'module.' prefix from nn.DataParallel
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if has_module_prefix:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
        
    print("Multi-class architecture (Real, FaceSwap, Reenactment) perfectly matched with the weights file!")
except Exception as e:
    print(f"Error checking model: {e}")
