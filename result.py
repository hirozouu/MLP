# module
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import model
import data

# const
num_inputs = 784 # input unit
num_outputs = 10 # output unit
num_layer = 3 # hidden layer
num_hidden = 32 # hidden unit
num_epochs = 20 # epochs

device = torch.device('cuda')

mynet = model.MLP(num_inputs,num_hidden,num_layer,num_outputs).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=mynet.parameters(), lr=0.1)

history_loss = []
history_eval = []
history_acc = []

dataset = data.HandwrittenDigit()

# 学習
for epoch in range(num_epochs):
    mynet.train()

    total_loss = 0.0 # 学習データに対する適合
    for i, (data, target) in enumerate(dataset.train_loader):
        data = data.view(-1, 28*28)

        optimizer.zero_grad()
        output = mynet(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.cpu().item()


    # 検証
    num_correct = 0
    num_data = 0

    mynet.eval()
    with torch.no_grad():
        eval_loss = 0.0 # テストデータに対する適合
        for i, (data, target) in enumerate(dataset.test_loader):
            data = data.view(-1, 28*28)
            output = mynet(data.to(device))
            loss = criterion(output, target.to(device))
            eval_loss = eval_loss + loss.cpu().item()
            num_correct = num_correct + output.cpu().argmax(dim=1).eq(target).sum()
            num_data = num_data + data.shape[0]

    history_loss.append(total_loss)
    history_eval.append(eval_loss)
    history_acc.append(num_correct.item()/num_data)
    print("{}/{} training loss: {}, evaluation loss: {}".format(epoch,num_epochs,total_loss,eval_loss))
    print("accuracy: {}/{}={}".format(num_correct, num_data,num_correct.item()/num_data))
    rnd=random.sample(range(len(target)),10)
    for i in range(10):
        print("(prediction: {}, truth: {}), ".format(output[rnd[i]].argmax().item(), target[rnd[i]].item()), end='')
        if(i==4 or i==9):
            print()
    print()

plt.figure()
plt.plot(range(1, num_epochs+1), history_loss, label='training loss')
plt.title('history')
plt.xlabel('epoch')
plt.legend()
plt.show()
#plt.savefig('total_loss.png')

plt.figure()
plt.plot(range(1, num_epochs+1), history_eval, label='evaluation loss')
plt.title('history')
plt.xlabel('epoch')
plt.legend()
plt.show()
#plt.savefig('eval_loss.png')

plt.figure()
plt.plot(range(1, num_epochs+1), history_acc, label='accurary')
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
