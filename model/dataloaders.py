import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def get_loader(pin_memory=True,num_workers = 8,batch_size = 256):
    """
    Return the dataloader for train,dev and test, along with number of Classes in dataset

    Args:
        pin_memory: (bool) speed up host to device transfer(load samples on CPU push to GPU on training)
        number_workers: (int) multi-process data loading
        batch_size: (int) load data in batches
    
    Returns:
        dataloaders: (DataLoader) train, test, and dev dataloaders
        num_classses: (int) number of different classes of faces in dataset

    """
    data_path = './data/face_verf/classification_data/'
    #data_path = 'C:/Users/bibek/OneDrive/Desktop/Course_UNL/dataset/classification_data'
    # Resize, Normalize and convert to tensor an Image of 32*32*3 according to model
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Train Dataset
    train_dataset = torchvision.datasets.ImageFolder(root=data_path+'train_data/', 
                                                 transform = transform)

    # Dev Dataset                                             
    dev_dataset = torchvision.datasets.ImageFolder(root=data_path+'val_data/', 
                                               transform = transform)
    
    # Test Dataset
    test_dataset = torchvision.datasets.ImageFolder(root=data_path+'test_data/', 
                                               transform = transform)
    
    # Train Loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               shuffle=True, num_workers=num_workers,pin_memory=True)

    
    # Dev Loader
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=num_workers, pin_memory=True)

    # Test Loader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, 
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
  

    return train_dataloader, dev_dataloader,test_dataloader, len(train_dataset.classes)                                     