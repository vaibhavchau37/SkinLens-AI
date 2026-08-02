from PIL import Image
import numpy as np

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    valid_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        ToTensorV2(),
    ])
except ImportError:
    from torchvision import transforms as T

    class TorchvisionTransform:
        def __init__(self):
            self.transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        def __call__(self, image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))

            image = image.convert("RGB")
            tensor = self.transform(image)
            return {"image": tensor}

    valid_transform = TorchvisionTransform()