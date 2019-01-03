import os
import pickle as pkl
import matplotlib.pyplot as plt

folder = '../hyper/initializer'

curves = []
files = os.listdir(folder)
for file in files:
    with open(os.path.join(folder, file), 'rb') as f:
        curves.append(pkl.load(f))
epoch = [x['step'] for x in curves]
loss = [x['loss'] for x in curves]
train_acc = [x['train_acc'] for x in curves]
test_acc = [x['test_acc'] for x in curves]
for x, y, label in zip(epoch, test_acc, files):
    plt.plot(x, y, label=label)
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.legend()
plt.show()
