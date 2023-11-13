>  ML2023Spring - HW01 相关信息：
>  
>  [课程主页](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
>  
>  [课程视频](https://www.bilibili.com/video/BV1TD4y137mP/?spm_id_from=333.337.search-card.all.click&vd_source=436107f586d66ab4fcf756c76eb96c35)
>  
>  [Kaggle link](https://www.kaggle.com/t/a339b77fa5214978bfb8dde62d3151fe)
>  
>  [Sample code](https://colab.research.google.com/drive/1BESEu-l3qrGRULoATuXnWasUNuUlVF1Z?fbclid=IwAR1FrjUsp4rTy5PPFV-aWq6IG_Z44mFT4VH5e1lIhlekFl7fAvxGRCTCyR0#scrollTo=QoWPUahCtoT6)
>  
>  [HW01 视频]( https://www.bilibili.com/video/BV1TD4y137mP/?p=14&share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390) 可以在做作业之前看一部分，我摸索完才发现视频有讲 Data Feature :(
>  
>  [HW01 PDF](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/HW01.pdf)
>  
>  
>P.S. 即便 kaggle 上的时间已经截止，你仍然可以在上面提交和查看分数。但需要注意的是：在 kaggle 截止日期前你应该选择两个结果进行最后的Private评分。
>  每年的数据集size和feature并不完全相同，但基本一致，过去的代码仍可用于新一年的 Homework

# 目录
* [任务目标（回归）](#任务目标回归)
* [性能指标（Metric）](#性能指标metric)
* [数据解析](#数据解析)
   * [数据下载](#数据下载)
* [Sample code 主体部分解析](#sample-code-主体部分解析)
   * [Some Utility Functions](#some-utility-functions)
   * [Dataset](#dataset)
   * [Neural Network Model](#neural-network-model)
   * [Feature Selection](#feature-selection)
   * [Training Loop](#training-loop)
* [Baselines](#baselines)
* [参考链接](#参考链接)

# 任务目标（回归）：

- COVID-19 daily cases prediction: COVID-19 每天的病例预测
- 训练/测试数据大小：3009/997（每一年的homework 可能不同）

# 性能指标（Metric）

- 均方误差 Mean Squared Error (MSE) 

# 数据解析

- covid_train.txt: 训练数据
- covid_test.txt: 测试数据

数据大体分为三个部分：id, states: 病例对应的地区, 以及其他数据
- id: sample 对应的序号。
- states: 对 sample 来说该项为 one-hot vector。从整个数据集上来看，每个地区的 sample 数量是均匀的，可以使用`pd.read_csv('./covid_train.csv').iloc[:,1:34].sum()`来查看，地区 sample 数量为 88/89。
- 其他数据: 这一部分最终应用在助教所给的 sample code 中的 select_feat。

    - Covid-like illness (5) 新冠症状

      - cli, ili ...

    - Behavier indicators (5) 行为表现

      - wearing_mask、travel_outside_state ... 是否戴口罩，出去旅游 ...

    - Belief indicators (2) 是否相信某种行为对防疫有效

      - belief_mask_effective, belief_distancing_effective. 相信戴口罩有效，相信保持距离有效。

    - Mental indicator (2) 心理表现

      - worried_catch_covid, worried_finance.  担心得到covid，担心经济状况

    - Environmental indicators (3) 环境表现

      - other_masked_public, other_distanced_public ... 周围的人是否大部分戴口罩，周围的人是否大部分保持距离 ...

    - Tested Positive Cases (1) 检测阳性病例，该项为模型的预测目标

      - **tested_positive (this is what we want to predict)** 单位为百分比，指有多少比例的人  

## 数据下载

> To use the Kaggle API, sign up for a Kaggle account at [https://www.kaggle.com](https://www.kaggle.com/). Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with `echo %HOMEPATH%`). You can define a shell environment variable `KAGGLE_CONFIG_DIR` to change this location to `$KAGGLE_CONFIG_DIR/kaggle.json` (on Windows it will be `%KAGGLE_CONFIG_DIR%\kaggle.json`).
>
> -\- [Official Kaggle API](https://github.com/Kaggle/kaggle-api)

`gdown` 的链接总是挂，可以考虑使用 `kaggle` 的 `api`，流程非常简单，替换<username>为你自己的用户名，`https://www.kaggle.com/<username>/account`，然后点击 `Create New API Token`，将下载下来的文件放去应该放的位置：

- Mac 和 Linux 放在 `~/.kaggle`
- Windows 放在 `C:\Users\<Windows-username>\.kaggle`

```bash
pip install kaggle
# 你需要先在 Kaggle -> Account -> Create New API Token 中下载 kaggle.json
# mv kaggle.json ~/.kaggle/kaggle.json
kaggle competitions download -c ml2023spring-hw1
unzip ml2023spring-hw1
```

# Sample code 主体部分解析

## Some Utility Functions

```python
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    # 使用确定的卷积算法 (A bool that, if True, causes cuDNN to only use deterministic convolution algorithms.)
    torch.backends.cudnn.deterministic = True	
    
    # 不对多个卷积算法进行基准测试和选择最优 (A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.)
    torch.backends.cudnn.benchmark = False	
    
    # 设置随机数种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
	# 用于评估模型（验证/测试）
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
    	# device (int, optional): if specified, all parameters will be copied to that device）     	                  
        x = x.to(device)	# 将数据 copy 到 device
        with torch.no_grad():	# 禁用梯度计算，以减少消耗                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   # detach() 创建一个不在计算图中的新张量，值相同
    preds = torch.cat(preds, dim=0).numpy()  # 连接 preds 
    return preds
```

## Dataset

```python
class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

	'''meth:`__getitem__`, supporting fetching a data sample for a given key.'''
    def __getitem__(self, idx):	# 自定义 dataset 的 idx 对应的 sample
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
```

\_\_getitem\_\_()实际应用于 dataloader 中，详细可见下图（图源自 [PyTorch Tutorial PDF](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/environment.pdf)）

![源自 Pytorch PDF](https://blogby.oss-cn-guangzhou.aliyuncs.com/20230314135050.png)

## Neural Network Model

这部分我做了简单的修改，以便于后续调参

```python
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure in hyper-parameter: 'config', be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, config['layer'][0]),
            nn.ReLU(),
            nn.Linear(config['layer'][0], config['layer'][1]),
            nn.ReLU(),
            nn.Linear(config['layer'][1], 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```

## Feature Selection

这部分可以使用 sklearn.feature_selection.SelectKBest 来进行特征选择。
具体代码如下（你可能需要传入 config）：

```python
from sklearn.feature_selection import SelectKBest, f_regression

k = config['k']	# 所要选择的特征数量
selector = SelectKBest(score_func=f_regression, k=k)
result = selector.fit(train_data[:, :-1], train_data[:,-1])
idx = np.argsort(result.scores_)[::-1]
feat_idx = list(np.sort(idx[:k]))
```

## Training Loop

```python
def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum']) 	# 设置 optimizer 为SGD
    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []	# 初始化空列表，用于记录训练误差

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)	# 让训练进度显示出来，可以去除这一行，然后将下面的 train_pbar 改成 train_loader（目的是尽量减少 jupyter notebook 的打印，因为如果这段代码在 kaggle 执行，在一定的输出后会报错: IOPub message rate exceeded...）

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)	# 等价于 model.forward(x)             
            loss = criterion(pred, y)	# 计算 pred 和 y 的均方误差
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        loss_record = []	# 初始化空列表，用于记录验证误差
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
```

# Baselines

根据作业 PDF 中的提示：

- Simple Baseline (1.96993)
  - 运行所给的 sample code。
- Medium Baseline (1.15678)
  - 特征选择，简单手动的选择你认为关联性较大的特征。
- Strong Baseline (0.92619)
  - 尝试不同的优化器（如：Adam）。
  - 应用 L2 正则化（SGD/Adam ... 优化器参数中的 weight_decay)
- Boss Baseline (0.81456)
  - 尝试更好的特征选择，可以使用 sklearn.feature_selection.SelectKBest。
  - 尝试不同的模型架构（调整 my_module.layers）
  - 调整其他超参数


# 参考链接

1. [PyTorch: What is the difference between tensor.cuda() and tensor.to(torch.device("cuda:0"))?](https://stackoverflow.com/questions/62907815/pytorch-what-is-the-difference-between-tensor-cuda-and-tensor-totorch-device)
1. [PyTorch Tutorial PDF](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/environment.pdf)




