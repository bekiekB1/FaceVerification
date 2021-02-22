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
