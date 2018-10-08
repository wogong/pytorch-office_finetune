"""Train classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

from utils.utils import save_model
from core.test import eval


def train_src(model, src_data_loader, tgt_data_loader_eval, device, params):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer

    parameter_list = [
        {
            "params": get_parameters(model.features, 'weight'),
            "lr": 0.001
        },
        {
            "params": get_parameters(model.features, 'bias'),
            "lr": 0.002
        },
        {
            "params": get_parameters(model.fc, 'weight'),
            "lr": 0.01
        },
        {
            "params": get_parameters(model.fc, 'bias'),
            "lr": 0.02
        },
    ]
    optimizer = optim.SGD(parameter_list, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    global_step = 0
    for epoch in range(params.num_epochs):
        for step, (images, labels) in enumerate(src_data_loader):
            model.train()
            global_step += 1
            adjust_learning_rate(optimizer, global_step)

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
            if (global_step % params.log_step == 0):
                print("Epoch [{:4d}] Step [{:4d}]: loss={:.5f}".format(
                    epoch + 1, global_step, loss.data.item()))

            # eval model on test set
            if (global_step % params.eval_step == 0):
                eval(model, src_data_loader, device)
                eval(model, tgt_data_loader_eval, device)

            # save model parameters
            if (global_step % params.save_step == 0):
                save_model(
                    model, params.src_dataset +
                    "-source-classifier-{}.pt".format(global_step), params)

        # end
        if (global_step > params.max_step):
            break

    # save final model
    save_model(model, params.src_dataset + "-source-classifier-final.pt",
               params)

    return model


def adjust_learning_rate(optimizer, global_step):
    lr_0 = 0.01
    gamma = 0.001
    power = 0.75
    lr = lr_0 / (1 + gamma * global_step)**power
    #print('lr in step {} is {}'.format(global_step, lr))
    optimizer.param_groups[0]['lr'] = lr * 0.1
    optimizer.param_groups[1]['lr'] = lr * 0.2
    optimizer.param_groups[2]['lr'] = lr * 1
    optimizer.param_groups[3]['lr'] = lr * 2


def get_parameters(module, flag):
    """ flag = 'weight' or 'bias'
    """
    for name, param in module.named_parameters():
        if flag in name:
            yield param