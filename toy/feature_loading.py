import torch
import torchvision.models as models

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, target_layer):
        super(ResNetFeatureExtractor, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        # Dissect the resnet up to the target_layer
        self.features = torch.nn.Sequential(*list(self.resnet.children())[:-2])
        
        
    def forward(self, x):
        print("1",x.shape)
        x = self.features(x)
        print("2",x.shape)
        return x

# Example: Extract features from layer 7 (layer counting starts from 0)
target_layer = 7
model = ResNetFeatureExtractor(target_layer)

# Dummy input tensor of the correct shape for ResNet50 (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 224, 224)
print(input_tensor.shape)
# Extract features
features = model(input_tensor)
print(features.shape)