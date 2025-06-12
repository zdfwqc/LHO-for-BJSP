import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
from  memory import Memory
from network.deepSetNet import DeepSetNet

def train_model(memory, model, num_epochs=200, batch_size=256, learning_rate=0.0002, testMemory=None,
                modelName = 'model', l2_lambda=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        all_features, all_targets = memory.sample(len(memory))
        indices = torch.randperm(len(memory))
        all_features = all_features[indices]
        all_targets = all_targets[indices]


        for i in range(0, len(memory) - batch_size, batch_size):
            batch_features = all_features[i:i + batch_size]
            batch_targets = all_targets[i:i + batch_size]
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets.unsqueeze(1))
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)
            loss = loss + l2_lambda * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1


        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)


        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
            if testMemory is not None and len(testMemory) > 0:
                model.eval()  # 设置为评估模式
                total_test_loss = 0
                with torch.no_grad():
                    for test_feature, test_target in testMemory.memory:
                        test_feature = torch.FloatTensor(np.array(test_feature)).unsqueeze(0)  # 添加batch维度
                        test_target = torch.FloatTensor([test_target])
                        test_prediction = model(test_feature)
                        test_loss = criterion(test_prediction, test_target.unsqueeze(1))
                        total_test_loss += test_loss.item()
                    avg_test_loss = total_test_loss / len(testMemory)
                    test_losses.append(avg_test_loss)
                    print(f'\033[91m测试集平均Loss: {avg_test_loss:.4f}\033[0m')  # 红色字体打印

                model.train()  # 恢复训练模式
    savePath = 'model/'+modelName+'.pth'
    torch.save(model.state_dict(), savePath)
    print('模型已保存到' + savePath)
    
    np.save('log/'+modelName+'_train_loss.npy',train_losses)
    np.save('log/'+modelName+'_test_loss.npy',test_losses)
    return model

if __name__ == '__main__':
    k = 7
    memory = Memory.load('trainData/TrainData_' + str(k) + '_5')
    memory.normalize()
    train,test = memory.split_memory(0.1)

    model = DeepSetNet(input_dim=15)
    train_model(model = model,memory=train, modelName='dsh75_NORM',testMemory=test, l2_lambda=0.01,num_epochs=1000)