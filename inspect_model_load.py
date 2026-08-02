import torch
from pathlib import Path
from model import BaselineModel

MODEL_PATH = Path('models/best_model.pth')
print('model file exists:', MODEL_PATH.exists())
ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
print('checkpoint type:', type(ck))
if isinstance(ck, dict):
    print('checkpoint keys:', list(ck.keys())[:20])
    if 'model_state_dict' in ck:
        state_dict = ck['model_state_dict']
        print('using model_state_dict')
    elif 'state_dict' in ck:
        state_dict = ck['state_dict']
        print('using state_dict')
    elif 'model' in ck and isinstance(ck['model'], dict):
        state_dict = ck['model']
        print('using nested model dict')
    else:
        state_dict = ck
else:
    state_dict = ck

print('state_dict length:', len(state_dict) if hasattr(state_dict, '__len__') else type(state_dict))
print('sample keys:', list(state_dict.keys())[:30])

model = BaselineModel(pretrained=False)
load_result = model.load_state_dict(state_dict, strict=False)
print('missing_keys:', load_result.missing_keys)
print('unexpected_keys:', load_result.unexpected_keys)
print('load_result:', load_result)
print('model state dict len:', len(model.state_dict()))
print('first model keys:', list(model.state_dict().keys())[:30])
