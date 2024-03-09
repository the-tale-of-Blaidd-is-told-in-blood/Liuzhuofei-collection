import torchvision
import torchvision.transforms
from torch.utils.data import DataLoader


dataset=torchvision.datasets.CIFAR10("./datasetCIF",True,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
testing_dataset=torchvision.datasets.CIFAR10("testing_dataset",False,
                                           transform=torchvision.transforms.ToTensor(),download=True)
testing_dataloader=DataLoader(testing_dataset,batch_size=64,shuffle=True,drop_last=True)