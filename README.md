# 李宏毅2023机器学习作业思路和代码分享

这里是我个人的 code 分享，所有的 code 最终均能达到 Boss baseline，希望能给你带来帮助。代码会一直更新到课程结束。

当前代码版本对应的是 2023 年春，其中 HW01 之前是 22 年版本，但我略作修改（修改文件路径）后便可直接用于现有版本，近年 Homework 的主题都一样，你完全可以从往年的优秀代码中学习知识。

现在所有的文件夹中都增加了 sample code 方便参考，HW_Simple_[score].ipynb 是对应 sample code 直接运行的结果，提供一个基准，从 HW3 开始代码分 Medium, Strong, Boss，方便快速索引改动。

描述文档中更新了各 Homework 的 Kaggle 邀请链接 (来自课程主页)，现在可以直接点击跳转参加，不会再遇到 limited-participation competition 的问题。

## 作业代码更新概述

- **HW01**：增加了 Adam, Momentum, Normalization, L2 regularization, Feature selection, K-fold cross validation 等相关代码，并使用了 Optuna 库进行了参数的自动搜寻。
- **HW02**：修改了原代码的小bug，增加了 tensorboard 和 scheduler 的使用，默认执行的是 strong baseline，你可以检查代码中的 TODO 选项去达成 Boss baseline（使用了BiLSTM）。
- **HW03**：代码根据 baseline 分为了三个部分(Medium, Strong,..)，方便大家查看，之后的代码都会遵循这一点。代码仅针对 sample code 的基础架构进行了训练，所以没有达到 boss baseline。目前代码的分数为0.85200，先上传给大家一点参考思路，闲暇后会重新进行 boss baseline 的跟进。
- **HW04**：
  - Medium: 通过 grid search 搜寻了几十个参数组合后修改了 transformer 模块中的参数。
  - Strong:  Medium 的基础上修改使用了 Conformer 论文中的 ConformerBlock 架构。
  - Boss: 添加了 hints 中描述的 Self-Attention Pooling 和 Additive Margin Softmax 模块，但实际上仅需要将 Strong 中 Classifier 部分的 pred_layer 同 Conformer 本身一样修改为单层的全连接层便可以非常轻易的达到 Boss baseline。
- **HW05**：提供中文版本，修正原sample code描述错误，因为 HW05/06 都是在 Judgeboi 上提交的，而非校内无法提交，所以这里后缀的分数是验证集上的分数。
  - Medium: 增加了学习率的调度和延长了训练的时间。
  - Strong: 将模型架构转变为了 Transformer，并根据 [Attention is all you need](https://arxiv.org/abs/1706.03762) 修改了模型的超参数。
  - Boss: 应用了 back-translation。
- **HW06**：编写了两个函数供大家测算 FID 和 AFD（在 Inference 之后），但需要注意，函数和 JudgeBoi 上的实现（模型不同）是不一致的，只是方便大家对比代码的改动效果。
  - Simple: 修改了 sample code 代码 Trainer 类中的 self.accelerator 部分以去除警告。
  - Medium: 数据增强，将 timesteps 增加至1000（同 [DDPM](https://arxiv.org/abs/2006.11239))）。
  - Strong: 调整超参数 channel 和 dim_mults，增加 cosine_beta_schedule() 和 sigmoid_beta_schedule()。
  - Boss: 使用 StyleGAN 进行图像生成。
- **HW07**：这门课程实际上引导学习了如何去微调一个 LLM 大语言模型来完成 QA 任务。我对所有的增加的模块做了参数上的对比实验，结果位于指导文档对应 Baseline 部分。
  - Medium: 提供 PyTorch和 Hugging Face 两种学习率调度方法，调整 `doc_stride` 参数提升模型表现。
  - Strong: 通过偏移答案的窗口位置来修复预处理阶段可能错误捕捉答案模式的问题，并对比展示四种预训练模型的训练结果。
  - Boss: 修复后处理部分 End < Start 的问题，增加梯度累积的使用，合并 dev 训练集，并更换模型参数更大的预训练模型。

...

## 克隆仓库

你可以使用下面的命令克隆仓库所有的文件到本地：

```bash
git clone https://github.com/Hoper-J/HUNG-YI_LEE_Machine-Learning_Homework.git
```

## 进一步学习

2024年生成式人工智能导论这门课程的中文引导和作业镜像版也已经制作完成，希望能够对你有所帮助：[LLM-Guide-and-Demos-zh_CN](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN)

