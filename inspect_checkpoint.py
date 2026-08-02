import torch
from pathlib import Path
MODEL_PATH = Path('models/best_model.pth')
print('path exists', MODEL_PATH.exists())
try:
    ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    print(type(ck))
    if isinstance(ck, dict):
        print('keys', list(ck.keys())[:20])
        if 'model_state_dict' in ck:
            print('model_state_dict keys', list(ck['model_state_dict'].keys())[:20])
        elif 'state_dict' in ck:
            print('state_dict keys', list(ck['state_dict'].keys())[:20])
        elif 'model' in ck and isinstance(ck['model'], dict):
            print('nested model keys', list(ck['model'].keys())[:20])
    else:
        print('non-dict checkpoint')
except Exception as e:
    import traceback
    traceback.print_exc()