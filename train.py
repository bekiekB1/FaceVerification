"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.model as net
import model.loss as loss
import model.dataloaders as data_loader
#from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/face_verf',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

def infer_classfication(net, loader,loss_claf,loss_verf):
    net.eval()
    running_loss = 0.0
    n = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs,labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            # forward + backward + optimize
            feature_vec,outputs = net(inputs)
            #loss = F.cross_entropy(outputs, labels)
            label_loss = loss_claf(outputs,labels)
            center_loss = loss_verf(feature_vec,labels)
            loss = label_loss + params.closs_weight * center_loss

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del inputs
            del labels

    acc = correct / total * 100
    avg_loss = running_loss / total
    return avg_loss, acc

def one_epoch(net,loader,optimizer_label,optimizer_center,claf_loss,criterion_closs,params):
  net.train()
  running_loss = 0.0
  n = 0
  correct = 0
  total = 0
  loss_avg = utils.RunningAverage()
  with tqdm(total=len(loader)) as t:
    for i,(inputs,labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #optimizer.zero_grad()
        optimizer_label.zero_grad()
        optimizer_center.zero_grad()
        feature_vec,outputs = net(inputs)
        label_loss = claf_loss(outputs,labels)
        center_loss = criterion_closs(feature_vec,labels)
        loss = label_loss + params.closs_weight * center_loss
        running_loss += loss.item()
        loss.backward()
        optimizer_label.step()

        for param in criterion_closs.parameters():
            param.grad.data *= (1. / params.closs_weight)
        optimizer_center.step()


        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        torch.cuda.empty_cache()
        loss_avg.update(loss.item())
        del inputs
        del labels
        del loss      
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        t.update()
  avg_loss = running_loss / total
  acc = correct / total *100
  return avg_loss,acc

def train_and_evaluate(model, train_dataloader, dev_dataloader, Adam_optimizer,optimizer_closs, loss_claf,loss_verf, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        dev_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        Adam_optimizer: (torch.optim) optimizer for parameters of model
        optimizer_closs: (CenterLoss) optimizer for Center Loss
        loss_claf: a function that takes batch_output and batch_labels and computes the loss for the batch
        loss_verf: CenterLoss 
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, Adam_optimizer,optimizer_closs)

    best_val_acc = 0.0
    train_losses = []
    valid_losses = []
    valid_acc = []
    auc_acc = []

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        #train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        avg_loss_t, acc_t = one_epoch(model,train_dataloader,Adam_optimizer,optimizer_closs,loss_claf,loss_verf,params)
        train_losses.append(avg_loss_t)
        logging.info('Epoch [%d], loss: %.8f, acc: %.4f' %
                (epoch + 1, avg_loss_t, acc_t))

        # Evaluate for one epoch on validation set
        #val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        avg_loss_v, acc_v = infer_classfication(model, dev_dataloader,loss_claf,loss_verf)
        valid_losses.append(avg_loss_v)
        valid_acc.append(acc_v)
        logging.info('[Classification valid] loss: %.8f, acc: %.4f\n\n' % (avg_loss_v, acc_v))
        #val_acc = val_metrics['accuracy']
        #is_best = val_acc >= best_val_acc
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'Adam_optimizer': Adam_optimizer.state_dict(),
                               'optimizer_closs': optimizer_closs.state_dict(),
                               'train_loss': train_losses,
                                'dev_loss':valid_losses,
                                'dev_acc': valid_acc,},
                              is_best=False,
                              checkpoint=model_dir)

        is_best = False

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        #utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if params.cuda else "cpu")

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dataloader, dev_dataloader,test_dataloader, num_classes = data_loader.get_loader()
    logging.info("- done.")

    # Define the model and optimizer
    model = net.resnet50(num_classes = num_classes,feat_dim = params.feat_dim).to(device)
    criterion_closs = loss.CenterLoss(num_classes, params.feat_dim, device)
    Adam_optimizer = torch.optim.Adam(model.parameters(),lr = params.learning_rate)
    optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=params.lr_cent)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Adam_optimizer,mode='min',patience=3)

    # fetch loss function
    loss_claf = loss.cross_entropy

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dataloader, dev_dataloader, Adam_optimizer,optimizer_closs, loss_claf,criterion_closs,  params, args.model_dir,
                       args.restore_file)
