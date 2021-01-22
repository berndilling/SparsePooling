import torch

from SparsePooling_pytorch.models import SparsePoolingModel, ClassificationModel
from SparsePooling_pytorch.utils import model_utils


def load_model(opt, num_GPU=None):

    model = SparsePoolingModel.SparsePoolingModel(opt)

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    # model, optimizer = model_utils.reload_weights(
    #     opt, model, optimizer, reload_model=reload_model
    # )

    return model #, optimizer

def load_classification_model(opt):
    if opt.in_channels == None:
        in_channels = 1024
    else:
        in_channels = opt.in_channels

    if opt.class_dataset == "stl10":
        num_classes = 10
    else:
        raise Exception("Invalid option")

    classification_model = ClassificationModel.ClassificationModel(
        in_channels=in_channels, num_classes=num_classes,
    ).to(opt.device)

    return classification_model
