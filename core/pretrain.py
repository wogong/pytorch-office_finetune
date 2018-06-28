"""Train classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

from utils.utils import save_model
from core.test import eval

def train_src(model, src_data_loader, src_data_loader_eval, tgt_data_loader, tgt_data_loader_eval, device, params):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    model.train()

    # setup criterion and optimizer
    parameter_list = [
        {"params": model.features.parameters(), "lr": 0.001},
        {"params": model.fc.parameters()},
        {"params": model.classifier.parameters()},
    ]
    optimizer = optim.SGD(parameter_list, lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        model.train()
        len_dataloader = len(src_data_loader)
        for step, (images, labels) in enumerate(src_data_loader):
            p = float(step + epoch * len_dataloader) / \
                params.num_epochs / len_dataloader
            adjust_learning_rate(optimizer, p)

            # make images and labels variable
            images = images.to(device)
            labels = labels.to(device)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = model(images)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{:4d}/{:4d}] Step [{:2d}/{:2d}]: loss={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len(src_data_loader),
                              loss.data.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step == 0):
            eval(model, src_data_loader_eval, device)
            eval(model, tgt_data_loader_eval, device)

        # save model parameters
        if ((epoch + 1) % params.save_step == 0):
            save_model(model, params.src_dataset + "-source-classifier-{}.pt".format(epoch + 1), params)

    # save final model
    save_model(model, params.src_dataset + "-source-classifier-final.pt", params)

    return model


def adjust_learning_rate(optimizer, p):
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha*p) ** beta
    for param_group in optimizer.param_groups[:1]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[1:]:
        param_group['lr'] = 10*lr