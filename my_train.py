from my_model import Mymodel
from torch import optim
from resource_data import dataloader,testing_dataloader
import torch
from torch import nn, optim
from torch.nn import Flatten
model=Mymodel()
loss_function=nn.CrossEntropyLoss()
learning_rate=0.005
min_testing_loss=100000.0
optim=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.7)
for epoch in range(50):
    total_loss=0.0

    # a=(epoch+1)/10
    # b=pow(0.5,a)
    # learning_rate=learning_rate*b
    for data in dataloader:
        imgs,targets=data
        weget=model(imgs)
        temp_loss=loss_function(weget,targets)  #通过调试可知，这里得到的是1维torch.tensor
        optim.zero_grad()
        temp_loss.backward()
        optim.step()
        total_loss+=temp_loss.item()


    print(f"模型第{epoch+1}轮在预测集上的交叉熵损失为：{total_loss}")
    with torch.no_grad():
        total_loss=0.0
        for data in testing_dataloader:
            input,value=data

            output=model(input)
            temp_loss=loss_function(output,value)
            total_loss+=temp_loss.item()
        print(f"模型第{epoch+1}轮在检测集上的交叉熵损失为：{total_loss}")
        if total_loss<min_testing_loss:
            min_testing_loss=total_loss
        else:
            torch.save(model, f"./state_data/best_data{epoch+1}")
            torch.save(model.state_dict(), f"./state_data/best_data_state_dict{epoch+1}")
