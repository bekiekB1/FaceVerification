import matplotlib.pyplot as plt

def plot_loss_acc(name,score):
    """
    Plots the loss or Acc against epochs
    
    Args:
        name: (string) Either Train loss, Dev Loss or Dev Accuracy
        score: (list) list of loss/accuray with epochs
    """
    plt.title(name)
    plt.xlabel('Epoch Number')
    plt.ylabel(name.split(sep=' ')[1])
    plt.plot(score)
    plt.savefig("graphs/"+name+".png")

def lerning_curve(train_loss, dev_loss):
    plt.figure(figsize=(10, 6))
    plt.plot( train_loss,"r-+", linewidth=3, label="Training loss")
    plt.plot(dev_loss, "b-", linewidth=2, label="Dev loss")
    plt.legend(loc="best", fontsize=14)   
    plt.xlabel("Epochs", fontsize=14) 
    plt.ylabel("Loss", fontsize=14) 
    plt.title("Learning Curve")
    plt.savefig("graphs/"+name+".png")
