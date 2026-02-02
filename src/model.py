import torch
import torch.nn as nn
import torchvision.models as models

def load_model(arch="resnet50", pretrained=True):
    """
    Loads pre-trained architectures for evaluation.
    Supported: 'resnet50', 'densenet121'
    """
    if arch == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    else:
        raise ValueError(f"Architecture {arch} not supported.")
    
    model.eval()
    return model

def adapt_batchnorm(model):
    """
    Implementation of Adaptive Batch Normalization (AdaBN).
    Configures BatchNorm layers to update running statistics 
    based on the test-time data distribution.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train() # Set to train mode to update statistics
            m.momentum = 0.1 # Standard momentum for adaptation
    return model