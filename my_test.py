import torch
from sklearn.metrics import accuracy_score
from resource_data import testing_dataloader
from my_model import Mymodel

def evaluate_model(model, dataloader):
    model.eval()  # 将模型设置为评估模式,会禁用在训练中使用的一些特定用于训练的功能
    predictions = []  #创建用于储存模型预测值的空列表
    targets = []   #创建用于储存真实标签的空列表
    with torch.no_grad():               ##确保此范围内不会有梯度计算，以加快计算速度
        for data in dataloader:
            imgs, labels = data
            outputs = model(imgs)
            _,predicted = torch.max(outputs, 1)     #加个横岗表示我们不接收第一个返回值，我们只关心第二个返回值
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    accuracy = accuracy_score(targets, predictions)
    return accuracy
model=Mymodel()
weight=torch.load("./state_data/best_data_state_dict16")
model.load_state_dict(weight)
# print(model)
acc=evaluate_model(model,testing_dataloader)
print(acc)
