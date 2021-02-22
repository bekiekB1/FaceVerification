"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.model as net
import model.loss as loss
#from train import infer_classfication
import model.dataloaders as data_loader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from model.metric import get_roc_auc_score,get_cosine_similarity,get_roc_curve
import pandas as pd
from graphs.plotter import plot_loss_acc

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/face_verf',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

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

class MyDatasetVerify(Dataset):
    # Create dataset for validation of verification tast
    def __init__(self, pairFN, imgFolderPath):
        self.imgFolderPath = imgFolderPath
        with open(pairFN) as f:
            self.pairList = [line.rstrip() for line in f]
    def __len__(self):
        return len(self.pairList)
    def __getitem__(self, idx):
        transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        items = self.pairList[idx].split()
        fn1, fn2 = items[0], items[1]
        img1 = Image.open(self.imgFolderPath + fn1)
        img2 = Image.open(self.imgFolderPath + fn2)
        img1 = transform(img1)
        img2 = transform(img2)
        #img1 = transforms.ToTensor()(img1)
        #img2 = transforms.ToTensor()(img2)
        if len(items) == 3: # validation
            return img1, img2, int(items[2])
        else: # test
            return img1, img2, -1
    def getPairList(self):
        return self.pairList

def testVerify(model, vLoader):
    model.eval()
    similarity = np.array([])
    true = np.array([])
    with torch.no_grad():
        for batch_idx, (imgs1, imgs2, targets) in enumerate(vLoader):
            imgs1, imgs2, targets = imgs1.float().to(device), imgs2.float().to(device), targets.float().to(device)
            # find cos similarity between embeddings
            facImbd_img1 = model(imgs1.float())[0]
            facImbd_img2 = model(imgs2.float())[0]
            sim = get_cosine_similarity(facImbd_img1, facImbd_img2) 
            similarity = np.concatenate((similarity, sim.cpu().numpy().reshape(-1)))
            true = np.concatenate((true, targets.cpu().numpy().reshape(-1)))
            del imgs1
            del imgs2
            del targets
    return similarity, true

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available
    device = torch.device("cuda" if params.cuda else "cpu")

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")


    # fetch dataloaders
    train_dataloaders, dev_dataloaders, test_dataloaders, num_classes = data_loader.get_loader()
                                                                            
    file_path = 'data/face_verf/verification_pairs_val.txt'
    file_path_test = 'data/face_verf/verification_pairs_test.txt'
    image_path = 'data/face_verf/'
    testVeriLabelName = "graphs/test_veri_labels.npy"
    testVeriLabelCSVfn = "graphs/test_veri_labels.csv"
    

    logging.info("- done.")

    # Define the model
    model = net.resnet50(num_classes = num_classes,feat_dim = params.feat_dim).to(device)
    #model = net.Net(params).cuda() if params.cuda else net.Net(params)

    criterion_closs = loss.CenterLoss(num_classes, params.feat_dim, device)
    Adam_optimizer = torch.optim.Adam(model.parameters(),lr = params.learning_rate)
    optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=params.lr_cent)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Adam_optimizer,mode='min',patience=3)
    #loss_fn = net.loss_fn
    #metrics = net.metrics
    loss_claf = loss.cross_entropy

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    checkpoint = utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)


    
    
    plot_loss_acc("Training Loss",checkpoint["train_loss"])
    plot_loss_acc("Dev Loss",checkpoint["dev_loss"])
    plot_loss_acc("Dev Acc",checkpoint["dev_acc"])


    
    # Evaluate
    #Good classifer indicates good feature extractor
    # Test the classifer in Test dataset 
    # Devloader is used for toy dataset
    avg_loss_test, acc_test = infer_classfication(model,test_dataloaders,loss_claf,criterion_closs)
    logging.info(f"*** Classifier <-> Feature Extractor\n [Test] Loss:{avg_loss_test} \t Accuracy:{acc_test} ")

    #Verification Validation
    vrf_val_dataset = MyDatasetVerify(file_path,image_path)

    verf_val_loader = torch.utils.data.DataLoader(vrf_val_dataset, batch_size=2, 
                                             shuffle=False, num_workers = params.num_workers)
	# Calculate simliarity score
    cosSim_valid, trueScore_valid = testVerify(model, verf_val_loader)
    # Report AUC
    get_roc_curve(trueScore_valid,cosSim_valid)
    auc = get_roc_auc_score(trueScore_valid, cosSim_valid)
    print("*** AUC: {} ***".format(auc))

    
    #Verification for test
    vrf_test_dataset = MyDatasetVerify(file_path_test,image_path)

    verf_test_loader = torch.utils.data.DataLoader(vrf_test_dataset, batch_size=2, 
                                             shuffle=False, num_workers = params.num_workers)
    
    cosSim_test, _ = testVerify(model, verf_test_loader)

	# Save predictied similarity
    cosSim_test = np.array(cosSim_test)
    np.save(testVeriLabelName, cosSim_test)
    trial = np.array(vrf_test_dataset.getPairList())
    df = pd.DataFrame({"trial" : trial, "score" : cosSim_test})
    df.to_csv(testVeriLabelCSVfn, index=False)

    

    '''
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    '''