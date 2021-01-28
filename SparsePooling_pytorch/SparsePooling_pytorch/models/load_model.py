import torch
import torch.nn as nn
import os

from SparsePooling_pytorch.models import SparsePoolingModel, ClassificationModel

def load_model(opt, num_GPU=None, reload_model=False):
    model = SparsePoolingModel.SparsePoolingModel(opt)

    model, num_GPU = distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    model = reload_weights(opt, model, reload_model=reload_model)

    return model

def reload_weights(opt, model, reload_model=False):
    # reload weights for investigation or training of downstream linear classifier
    if reload_model:
        print("Loading weights from ", opt.model_path)
        for idx, layer in enumerate(model.module.layers):
            model.module.layers[idx].load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path,
                        "model_{}_{}.ckpt".format(idx, opt.model_num),
                    ),
                        map_location=opt.device.type,
                )
            )
    else:
        print("Randomly initialized model")
    
    return model

def distribute_over_GPUs(opt, model, num_GPU):
    if opt.device.type != "cpu":
        if num_GPU is None:
            model = nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You can't use more GPUs than you have."
            model = nn.DataParallel(model, device_ids=list(range(num_GPU)))
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.batch_size_multiGPU = opt.batch_size

    model = model.to(opt.device)
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU

def load_classification_model(opt):
    if opt.in_channels == None:
        in_channels = 1024
    else:
        in_channels = opt.in_channels

    if opt.dataset_class == "stl10":
        num_classes = 10
    else:
        raise Exception("Invalid option")

    classification_model = ClassificationModel.ClassificationModel(
        in_channels=in_channels, num_classes=num_classes,
    ).to(opt.device)

    return classification_model
