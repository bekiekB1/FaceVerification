import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def get_cosine_similarity(face_imbedding1,face_imbedding2):
    return F.cosine_similarity(face_imbedding1,face_imbedding2)

def get_roc_curve(true_score,predict_score):
    fpr, tpr, thresholds = roc_curve(true_score, predict_score)
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 6))
    plot_roc_curve(fpr, tpr)
    plt.savefig('graphs/roc.png')


def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, color='darkorange', linewidth=8, label=label) 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axis([0, 1, 0, 1])
    plt.title('ROC Curve (Train Data)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def get_roc_auc_score(true_score,predict_score):
    return roc_auc_score(true_score, predict_score)