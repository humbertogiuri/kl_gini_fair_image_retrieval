import torch
from torchvision import transforms, models
from torch.nn import Identity
import os
import pandas as pd
import cv2


class FeatureExtractor:
    def __init__(self, model_path=None):
        # Check for GPU availability
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f'Using device: {self.device}')

        if model_path == None:
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT
            )

        # Load the model
        else:
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load(model_path))

        # Replace the last layer (fc layer) with an AdaptiveAvgPool2d to get the feature vector
        self.model.fc = Identity()
        self.model.eval()

        # Define the standard ResNet transformations
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(self, img):
        """Extract feature vector from an image"""
        # Preprocess the image
        img = self.preprocess(img).unsqueeze(0)  # Add a batch dimension
        img = img.to(
            self.device
        )  # Move the image tensor to the same device as the model

        # Extract features
        with torch.no_grad():  # No need to track gradients for feature extraction
            features = self.model(img)

        # Remove unnecessary dimensions and convert to numpy
        features_np = features.squeeze().cpu().numpy()
        return features_np


# Example usage
if __name__ == '__main__':
    extractor = FeatureExtractor()
    path_train = './NDB-ORIGINAL/train'

    classes = os.listdir(path_train)

    dict_results = {'class': list(), 'features': list(), 'name': list()}

    for classe in classes:
        file_names = os.listdir(f'{path_train}/{classe}')

        for file_name in file_names:
            print(f'Processing -> {classe}/{file_name}')
            # Load an image with OpenCV
            image_path = f'{path_train}/{classe}/{file_name}'
            img = cv2.imread(image_path)
            img = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB
            )  # Convert BGR (OpenCV format) to RGB

            # Extract features
            features = extractor.extract_features(img)

            dict_results['class'].append(classe)
            dict_results['features'].append(features)
            dict_results['name'].append(file_name)

    df = pd.DataFrame(dict_results)
    df.to_hdf('data_ndb.h5', key='dframe')
