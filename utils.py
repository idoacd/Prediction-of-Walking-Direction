#utils.py

import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt 

from generate_data import directions

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = '14'
matplotlib.rcParams['figure.figsize'] = [10., 5.]
matplotlib.rcParams['figure.dpi'] = 120

def process_travels(travels):
    travels_tensor = torch.tensor(travels)
    travels_tensor = travels_tensor.reshape(-1, travels_tensor.size()[0], travels_tensor.size()[1])
    return travels_tensor.double()

def process_targets(targets, n_classes):
    targets_tensor = torch.tensor(targets)
    targets_tensor = F.one_hot(targets_tensor, num_classes=n_classes)
    targets_tensor = targets_tensor.reshape(-1, targets_tensor.size()[0], targets_tensor.size()[1])
    return targets_tensor.double()

def get_batch(train_travels, train_targets, indices, n_classes):
    batch_x = process_travels(train_travels[indices])
    batch_y = process_targets(train_targets[indices], n_classes)
    return batch_x, batch_y

def sum_errors(test_travels, test_targets, criterion, model, n_classes):
    test_loss = 0
    test_preds = []
    for i in range(len(test_travels)):
        travel_tensor = process_travels(test_travels[i])
        pred = model(travel_tensor)
        test_preds.append(torch.argmax(pred[0], dim=1))
        target_tensor = process_targets(test_targets[i], n_classes)
        loss = criterion(pred, target_tensor).item()
        test_loss += loss
    return test_loss/len(test_travels), test_preds

def draw(x1, x2, color1, color2):
    plt.figure()
    plt.title('Predict Next Step', fontsize=16)
    plt.plot(range(len(x1)), x1, color1, marker='.', linestyle='None')
    plt.plot(range(len(x2)), x2, color2, marker='.', linestyle='None')
    plt.yticks(list(directions.values()), list(directions.keys()))
    plt.xlabel('Walking Step')
    plt.show()

def draw_train_test_errors(losses):
    plt.figure()
    plt.title('Train and Test Loss vs. Number of Epochs')
    plt.plot(range(len(losses)), losses[:, 0])
    plt.plot(range(len(losses)), losses[:, 1])
    plt.legend(['Train Loss', 'Test Loss'])
    plt.show()