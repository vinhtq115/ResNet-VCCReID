from config import CONFIG
from models.classifier import Classifier, NormalizedClassifier
from models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50
from models.resnet50 import ResNet
from models.resnet50_attn import ResNetAttn


__factory = {
    'c2dres50': C2DResNet50,
    'resnet50': ResNet,
    'resnet50_attn': ResNetAttn,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}


def build_models(config: CONFIG, num_ids: int = 150, num_clothes: int = 473, train=True):

    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        model = __factory[config.MODEL.NAME](config)

    if train:
        id_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM,
                                                    num_classes=num_ids)

        clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.APP_FEATURE_DIM, num_classes=num_clothes)

        return model, id_classifier, clothes_classifier
    else:
        return model
