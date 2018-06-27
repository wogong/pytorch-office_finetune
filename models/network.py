import torch.nn as nn
from torchvision import models
from .alexnet import alexnet

class AlexModel(nn.Module):
    """ AlexNet pretrained on imagenet for Office dataset"""

    def __init__(self):
        super(AlexModel, self).__init__()
        self.restored = False
        model_alexnet = models.alexnet(pretrained=True)

        self.features = model_alexnet.features

        self.fc = nn.Sequential()
        for i in range(6):
            self.fc.add_module("classifier" + str(i),
                               model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features  # 4096

        self.classifier = nn.Sequential(
            nn.Linear(4096, 31),
        )

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 227, 227)
        feature = self.features(input_data)
        feature = feature.view(-1, 256*6*6)
        fc = self.fc(feature)

        class_output = self.classifier(fc)

        return class_output

class AlexModel_LRN(nn.Module):
    """ AlexNet pretrained on imagenet for Office dataset"""

    def __init__(self):
        super(AlexModel_LRN, self).__init__()
        self.restored = False
        model_alexnet = alexnet(pretrained=True)

        self.features = model_alexnet.features

        self.fc = nn.Sequential()
        for i in range(6):
            self.fc.add_module("classifier" + str(i),
                               model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features  # 4096

        self.classifier = nn.Sequential(
            nn.Linear(4096, 31),
        )

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 227, 227)
        feature = self.features(input_data)
        feature = feature.view(-1, 256*6*6)
        fc = self.fc(feature)

        class_output = self.classifier(fc)

        return class_output
