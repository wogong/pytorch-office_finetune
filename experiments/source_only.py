import os
import torch
import sys
sys.path.append('../')
from models.network import AlexModel, AlexModel_LRN
from utils.utils import get_data_loader, init_random_seed
from core.pretrain import train_src
from core.test import eval

class Config(object):
    # params for dataset and data loader
    dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
    model_root = os.path.expanduser(
        os.path.join('~', 'Models', 'pytorch-alexnet_office_baseline'))
    batch_size = 64

    # params for source dataset
    src_dataset = "amazon31"

    # params for target dataset
    tgt_dataset = "webcam31"

    # params for setting up models
    model_trained = False
    model_restore = os.path.join(
        model_root, src_dataset + "-final.pt")

    # params for training network
    num_gpu = 1
    num_epochs = 500
    log_step = 10  # iter
    eval_step = 1  # epoch
    save_step = 5000
    manual_seed = None

params = Config()

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # init device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    src_data_loader = get_data_loader(
        params.src_dataset, dataset_root=params.dataset_root, batch_size=params.batch_size, train=True)
    tgt_data_loader = get_data_loader(
        params.tgt_dataset, dataset_root=params.dataset_root, batch_size=params.batch_size, train=True)

    # load models
    #model = AlexModel_LRN().to(device)
    model = AlexModel().to(device)

    # training model
    print("training model")
    if not (model.restored and params.model_trained):
        model = train_src(model, src_data_loader, tgt_data_loader, device, params)

    # eval trained model
    print("eval trained model")
    eval(model, tgt_data_loader, device)

    # end
    print("done")
