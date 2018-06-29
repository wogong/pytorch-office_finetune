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

        self.fc.add_module("final", nn.Linear(4096, 31))

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        feature = self.features(input_data)
        feature = feature.view(-1, 256*6*6)
        class_output = self.fc(feature)
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

class ResModel(nn.Module):
    def __init__(self):
        super(ResModel, self).__init__()
        self.restored = False

        model_resnet50 = models.resnet50(pretrained=True)

        self.features = nn.Sequential(
            model_resnet50.conv1,
            model_resnet50.bn1,
            model_resnet50.relu,
            model_resnet50.maxpool,
            model_resnet50.layer1,
            model_resnet50.layer2,
            model_resnet50.layer3,
            model_resnet50.layer4,
            model_resnet50.avgpool,
        )
        self.__in_features = model_resnet50.fc.in_features
        self.fc = nn.Linear(self.__in_features, 31)

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x