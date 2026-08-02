import importlib
mods = ['torch','torchvision','cv2','albumentations','streamlit','numpy']
for name in mods:
    try:
        m = importlib.import_module(name)
        print(name, 'OK', getattr(m, '__version__', ''))
    except Exception as e:
        print(name, 'ERR', repr(e))
