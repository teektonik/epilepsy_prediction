from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_outputs = []
    all_targets = []
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.float()  # Ensure the inputs are float
        targets = targets.long()  # Ensure the targets are long
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_outputs.append(outputs.detach().numpy())
        all_targets.append(targets.numpy())
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    f1 = f1_score(all_targets, np.argmax(all_outputs, axis=1))
    accuracy = accuracy_score(all_targets, np.argmax(all_outputs, axis=1))
    recall = recall_score(all_targets, np.argmax(all_outputs, axis=1))
    print(f'Train Loss: {total_loss / len(train_loader):.4f}, 
          Train F1: {f1:.4f}, 
          Train Accuracy: {accuracy:.4f}, 
          Train Recall: {recall:.4f}')
    fpr, tpr, _ = roc_curve(all_targets, all_outputs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return total_loss / len(train_loader), roc_auc, f1, accuracy, recall
