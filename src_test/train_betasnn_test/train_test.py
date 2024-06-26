from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
print(sys.path)
import tonic

from src.beta_csnn import BetaCSNN 


datapath=Path(__file__).parent.parent.parent/"survey/train_dvs/data"

dataset = tonic.datasets.NMNIST(save_to=datapath, train=True)
events, target = dataset[0]
print(events)

import tonic.transforms as transforms

sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                      transforms.ToFrame(sensor_size=sensor_size, 
                                                         time_window=1000)
                                     ])

trainset = tonic.datasets.NMNIST(save_to=datapath, transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to=datapath, transform=frame_transform, train=False)


# from tonic import MemoryCachedDataset
# cached_trainset = MemoryCachedDataset(trainset) #メモリキャッシュ (やってもいいかもね)
# cached_dataloader = DataLoader(cached_trainset)


#>> データ準備 >>
import torch
import torchvision
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.Resize((32, 32)),
                                      torchvision.transforms.RandomRotation([-10,10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

# no augmentations for the testset
cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

batch_size = 64 #timestepを311とすると64が限界
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))
#>> データ準備 >>


#>> モデル準備 >>
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn

device_id=0
device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.atan()
beta = 0.5


def load_yaml_as_dict(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    """
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


#  Initialize Network
class Net(nn.Module):
    def __init__(self,conf):
        super(Net, self).__init__()

        self.beta_snn=BetaCSNN(conf=conf,spike_grad=spike_grad,device=device).to(device)

        self.snn = nn.Sequential(nn.Conv2d(16, 32, 3,1,1),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3,1,1),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)
        
    def forward(self,spike,is_train_beta=False):
        """
        :param spike: [timesteps x batch x 2 x h x w]
        """
        T=spike.shape[0]
        spike,_=self.beta_snn.forward(spike,is_train_beta)

        utils.reset(self.snn)
        out_spike=[]
        for t in range(T):
            sp,_=self.snn(spike[t])
            out_spike.append(sp)
        out_spike=torch.stack(out_spike)

        return out_spike
    
conf=load_yaml_as_dict(Path(__file__).parent/"conf.yml")
print(conf)
net=Net(conf)
#>> モデル準備 >>


#>> 学習 >>
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
num_epochs = 10
num_iters = 50

loss_hist = []
acc_hist = []

# training loop

try:
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
          data = data.to(device) #[timestep x batch x channel x height x width], [311, 128, 2, 34, 34]時間方向デカすぎ
          targets = targets.to(device) #[batch]

          net.train()
          spk_rec = net.forward(data,is_train_beta=True)
          loss_val:torch.Tensor = loss_fn(spk_rec, targets)

          # Gradient calculation + weight update
          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()

          # Store loss history for future plotting
          loss_hist.append(loss_val.item())
  
          print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

          acc = SF.accuracy_rate(spk_rec, targets) 
          acc_hist.append(acc)
          print(f"Accuracy: {acc * 100:.2f}%\n")

          # This will end training after 50 iterations by default
          if i == num_iters:
              break
except Exception as e:
    print(e)
finally:

  import matplotlib.pyplot as plt
  from pathlib import Path

  # Plot Loss
  fig = plt.figure(facecolor="w")
  plt.plot(acc_hist)
  plt.title("Train Set Accuracy")
  plt.xlabel("Iteration")
  plt.ylabel("Accuracy")
  plt.savefig((Path(__file__).parent / "train_accuracy.png").resolve())