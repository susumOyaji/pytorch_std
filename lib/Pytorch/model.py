import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch

import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


'''
具体的なサンプルデータを生成
'''

sample = torch.randn(1, 10)
print(sample)

'''
nn.Sequentialを使用する書き方
'''

'''
nn.Sequentialの実装
'''
# Sequentialの書き方
model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
            )

print(model)

'''
nn.Sequentialの順伝搬
'''
#ネットワークが定義できたので、先ほどのサンプルデータを順伝搬させ、動作を確認しましょう。
y = model(sample)
print(y)


'''
ネットワークをモジュール化する書き方（nn.Moduleを継承）
'''
'''
ネットワークのモジュール化を実装
 

基本的な書き方は、nn.Moduleクラスを継承し、クラスに対して以下を定義します。

__init__ : コンストラクタでネットワークを定義する
forwardメソッド : forwardメソッドで順伝搬を定義する
 
'''

# ネットワークのモジュール化
class Model(nn.Module):
    def __init__(self, input):
        super(Model, self).__init__()
        
        # ネットワークを定義
        self.linear1 = nn.Linear(input, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    # 順伝搬を定義
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

#これで、Sequentialで定義したネットワークと同じネットワークを定義できます。


'''
自作定義したネットワークの順伝搬
'''

# instance化
model = Model(input=10)

# 順伝搬
y = model(sample)
print(y)


#summary(model,10)
'''
自作モジュールを複数使用して新たな自作モジュールを作成

自作モジュールを作成するメリットを理解していただけるように、簡単な例ですが、上記のネットワークを二つの自作モジュールを使用して構築します。

まずは、活性化関数も含めた各層の処理を以下のコードでモジュール化します。
'''

# Custum Layer
class CustomLayer(nn.Module):
    def __init__(self, input, output):
        super(CustomLayer, self).__init__()
        self.linear=nn.Linear(input, output)
        self.relu=nn.ReLU()
    def forward(self, x):
        x=self.linear(x)
        x=self.relu(x)
        return x
 

#さらに、この自作モジュールを使用して、これまで定義したネットワークと同じネットワークを作成します。

# ネットワークをモジュール化
class Model(nn.Module):
    def __init__(self, input):
        super(Model, self).__init__()
        self.custom1 = CustomLayer(input, 32)
        self.custom2 = CustomLayer(32, 16)
        self.linear3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.custom1(x)
        x = self.custom2(x)
        x = self.linear3(x)
        return x
 


#順伝搬を実装してみます。

model = Model(input=10)
y = model(sample)
print(y)

# torchsummaryを使った可視化
#summary(model,32)


model_ft = models.resnet18(pretrained = True)
print(model_ft)
summary(model_ft)