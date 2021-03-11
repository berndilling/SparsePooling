# adapted from https://github.com/loeweX/Greedy_InfoMax

import torch
import numpy as np
import time
import os

## own modules
from SparsePooling_pytorch.models import SparsePoolingModel, ClassificationModel, load_model
from SparsePooling_pytorch.arg_parser import arg_parser
from SparsePooling_pytorch.utils import logger, utils
from SparsePooling_pytorch.data import get_dataloader
from IPython import embed

def train_logistic_regression(opt, context_model, classification_model, train_loader):
    total_step = len(train_loader)
    classification_model.train()

    starttime = time.time()

    if opt.create_hidden_representation:
        hidden_reps = []
    for epoch in range(opt.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0
        for step, (img, target) in enumerate(train_loader):

            classification_model.zero_grad()

            if opt.create_hidden_representation and epoch > 0:
                with torch.no_grad():
                    z = hidden_reps[step]
                z = z.detach()
            else:
                model_input = img.to(opt.device)

                if opt.end_to_end_supervised:  # end-to-end supervised training
                    z, _ = context_model(model_input)
                else:
                    with torch.no_grad():
                        z, _ = context_model(model_input, up_to_layer=opt.class_from_layer)
                    z = z.detach() #double security that no gradients go to representation learning part of model
                
                if opt.create_hidden_representation and epoch == 0:
                    # pool here already to save memory on GPU, pooling is done anyway in classification_model
                    hidden_reps.append(torch.nn.functional.adaptive_avg_pool2d(z, 1).clone().detach()) 

            prediction = classification_model(z).to(opt.device)

            target = target.to(opt.device)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5

            sample_loss = loss.item()
            loss_epoch += sample_loss

            if step % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                        acc1,
                        acc5,
                        sample_loss,
                    )
                )
                starttime = time.time()

        if opt.validate:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, _ , val_loss = test_logistic_regression(
                opt, context_model, classification_model, test_loader
            )
            logs.append_val_loss([val_loss])

        print("Overall accuracy for this epoch: ", epoch_acc1 / total_step)
        # logs.append_train_loss([loss_epoch / total_step])
        logs.create_log(
            context_model,
            epoch=epoch,
            classification_model=classification_model,
            accuracy=epoch_acc1 / total_step,
            acc5=epoch_acc5 / total_step,
        )


def test_logistic_regression(opt, context_model, classification_model, test_loader):
    total_step = len(test_loader)
    context_model.eval()
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    for step, (img, target) in enumerate(test_loader):

        model_input = img.to(opt.device)

        if opt.end_to_end_supervised:  # end-to-end supervised training
                z, _ = context_model(model_input)
        else:
            with torch.no_grad():
                z, _ = context_model(model_input, up_to_layer=opt.class_from_layer)

        z = z.detach()

        prediction = classification_model(z)

        target = target.to(opt.device)
        loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        epoch_acc1 += acc1
        epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

        if step % 10 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc5, sample_loss
                )
            )
            starttime = time.time()

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step

if __name__ == "__main__":
    opt = arg_parser.parse_args()
    
    opt.classifying = True
    
    add_path_var = "classifier"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    # load pretrained model unless training whole model end-to-end
    if opt.end_to_end_supervised:
        context_model = load_model.load_model(opt)
        context_model.module.set_update_params(update_model = True, update_BP = True, update_SC_SFA = False) # switch plasticity on for BP layer, off for SC/SFA layers
    else:
        context_model = load_model.load_model(opt, reload_model=True)
        context_model.module.update_params = False # switching plasticity off


    if opt.class_from_layer==-1:
        print("CAREFUL! Training classifier directly on input image! Model is ignored and returns the (flattened) input images!")

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader_class(opt)

    classification_model = load_model.load_classification_model(opt)

    if opt.end_to_end_supervised: # end-to-end supervised training
        params = list(context_model.parameters()) + list(classification_model.parameters())
        #for p in params:
        #    print(p.shape)
    else:
        params = classification_model.parameters()

    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.CrossEntropyLoss()

    logs = logger.Logger(opt)

    try:
        # Train the model
        print("Training classifier on layer number (None=last layer): ", opt.class_from_layer)
        train_logistic_regression(opt, context_model, classification_model, train_loader)

        # Test the model
        acc1, acc5, _ = test_logistic_regression(
            opt, context_model, classification_model, test_loader
        )

    except KeyboardInterrupt:
        print("Training got interrupted")

    logs.create_log(
        context_model,
        classification_model=classification_model,
        accuracy=acc1,
        acc5=acc5,
        final_test=True,
    )
    torch.save(
        context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
    )

    np.save(os.path.join(opt.model_path, "classification_results_values_layer_"+str(opt.class_from_layer)+".npy"), 
            np.array([acc1, acc5]))
    L = ["Classification from layer: "+str(opt.class_from_layer)+"\n",
        "Test top1 classification accuracy: "+str(acc1)+"\n",
        "Test top5 classification accuracy: "+str(acc5)+"\n"]
    f = open(os.path.join(opt.model_path, "classification_results_layer_"+str(opt.class_from_layer)+".txt"), "w")
    f.writelines(L)
    f.close()