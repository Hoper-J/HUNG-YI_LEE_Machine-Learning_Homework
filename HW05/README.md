>  ML2023Spring - HW4 相关信息：
>
>  [课程主页](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
>
>  [课程视频](https://www.bilibili.com/video/BV1TD4y137mP/?spm_id_from=333.337.search-card.all.click&vd_source=436107f586d66ab4fcf756c76eb96c35)
>
>  [Sample code](https://colab.research.google.com/drive/17plKxw_Fm94E0SYrGMJe8T0BJluTBNb6)
>
>  [HW05 视频]( https://www.bilibili.com/video/BV1TD4y137mP/?p=40&share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390&t=6567)
>
>  [HW05 PDF](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/HW05.pdf)
>
>  运行日志: [wandb]( https://wandb.ai/hoper-hw/hw5.seq2seq/runs/ej174j09)
>
>  P.S. HW05/06 是在 Judgeboi 上提交的，完全遵循 hint 就可以达到预期效果。
>
>  因为无法在 Judgeboi 上提交，所以 HW05/06 代码仓库中展示的是在验证集上的分数。
>
>  每年的数据集 size 和 feature 并不完全相同，但基本一致，过去的代码仍可用于新一年的 Homework。
>
>  仓库中 HW05 的代码分成了英文 EN 和中文 ZH 两个版本。
>
>  （碎碎念：翻译比较麻烦，所以之后的 Homework 代码暂只有英文版本）


* [任务目标（seq2seq）](#任务目标seq2seq)
* [性能指标（BLEU）](#性能指标bleu)
* [数据解析](#数据解析)
* [Baselines](#baselines)
   * [Simple baseline (15.05)](#simple-baseline-1505)
   * [Medium baseline (18.44)](#medium-baseline-1844)
   * [Strong baseline (23.57)](#strong-baseline-2357)
   * [Boss baseline (30.08)](#boss-baseline-3008)
* [Gradescope](#gradescope)
   * [Visualize Positional Embedding](#visualize-positional-embedding)
   * [Clipping Gradient Norm](#clipping-gradient-norm)


# 任务目标（seq2seq）

- Machine translation 机器翻译，**英译中**

# 性能指标（BLEU）

> 参考链接：
>
> [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)
>
> [Foundations of NLP Explained — Bleu Score and WER Metrics](https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b)

**BLEU（Bilingual Evaluation Understudy）**  双语评估替换

公式：$ \text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n log\ p_n\right)^{\frac{1}{N}}$

首先要明确两个概念

1. **N-gram** 
   用来描述句子中的一组 n 个连续的单词。比如，"Thank you so much" 中的 n-grams:

   - 1-gram: "Thank", "you", "so", "much"
   - 2-gram: "Thank you", "you so", "so much"
   - 3-gram: "Thank you so", "you so much"
   - 4-gram: "Thank you so much"

   需要注意的一点是，n-gram 中的单词是按顺序排列的，所以 "so much Thank you" 不是一个有效的 4-gram。

2. **精确度（Precision）**
   精确度是 Candidate text 中与 Reference text 相同的单词数占总单词数的比例。 具体公式如下：
   $ \text{Precision} = \frac{\text{Number of overlapping words}}{\text{Total number of words in candidate text}} $
   比如：
   Candidate: <u>Thank you so much</u>, Chris
   Reference: <u>Thank you so much</u>, my brother
   这里相同的单词数为4，总单词数为5，所以 $\text{Precision} = \frac{{4}}{{5}}$
   但存在一个问题：

   - **Repetition** 重复

     Candidate: <u>Thank Thank Thank</u>
     Reference: <u>Thank</u> you so much, my brother

     此时的 $\text{Precision} = \frac{{3}}{{3}}$

**解决方法：Modified Precision**

很简单的思想，就是匹配过的不再进行匹配。

Candidate: <u>Thank</u> Thank Thank
Reference: <u>Thank</u> you so much, my brother

$\text{Precision}_1 = \frac{{1}}{{3}}$

- 具体计算如下：

  $Count_{clip} = \min(Count,\ Max\_Ref\_Count)=\min(3,\ 1)=1$
   $ p_n = \frac{\sum_{\text{n-gram}} Count_{clip}}{\sum_{\text{n-gram}} Count} = \frac{1}{3}$

现在还存在一个问题：**译文过短**

Candidate: <u>Thank you</u>
Reference: <u>Thank you</u> so much, my brother

$p_1 = \frac{{2}}{{2}} = 1$

这里引出了 **brevity penalty**，这是一个惩罚因子，公式如下：

$BP = \begin{cases} 1& \text{if}\ c>r\\ e^{1-\frac{r}{c}}& \text{if}\ c \leq r  \end{cases}$

其中 c 是 candidate 的长度，r 是 reference 的长度。

当候选译文的长度 c 等于参考译文的长度 r 的时候，BP = 1，当候选翻译的文本长度较短的时候，用 $e^{1-\frac{r}{c}}$ 作为 BP 值。

回到原来的公式：$ \text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n log\ p_n\right)^{\frac{1}{N}}$，汇总一下符号定义：

- $BP$ 文本长度的惩罚因子
- $N$ n-gram 中 n 的最大值，作业中设置为 4。
- $w_n$ 权重
- $p_n$ n-gram 的精度 (precision)

# 数据解析

- Paired data
  - TED2020: 演讲
    - Raw: 400,726 (sentences)
    - Processed: 394, 052 (sentences)
  - 英文和中文两个版本
- Monolingual data
  - 只有中文版本的 TED 演讲数据

# Baselines

> 这里存在一个问题，就是HW05是在 Judgeboi 上进行提交的，所以没办法获取最终的分数，所以简单的使用 simple baseline 对应的 validate BLEU 来做个映射。
>
> 因为有 EN / ZH 两个版本，对于每个 hint 我会给出代码的修改位置方便大家索引。

## Simple baseline (15.05)

- 运行所给的 sample code

## Medium baseline (18.44)

- 增加学习率的调度 (`Optimizer: Adam + lr scheduling` / `优化器: Adam + 学习率调度`)
- 训练得更久 (`Configuration for experiments` / `实验配置`)
  这里根据预估的时间，可以简单的将 epoch 设置为原来的两倍。

## Strong baseline (23.57)

- 将模型架构转变为 Transformer (`Model Initialization` / `模型初始化`)
- 调整超参数 (`Architecture Related Configuration` / `架构相关配置`)
  这里需要参考 [Attention is all you need](https://arxiv.org/abs/1706.03762) 论文中 table 3 的 transformer-base 超参数设置。
  ![image-20231115135033382](https://blogby.oss-cn-guangzhou.aliyuncs.com/20231115135033.png)

你可以仅遵循 sample code 的注释，将 encoder_layer 和 decoder_layer 改为 4（简单的将这一个改动称之为 transformer_4layer），此时模型的参数数量会和之前的 RNN 差不多，在 max_epoch =30 的情况下，Bleu 可以达到 23.59。

[代码仓库](https://github.com/Hoper-J/HUNG-YI_LEE_Machine-Learning_Homework)中分享的 Strong 代码完全遵循了 transformer-base 的超参数设置，此时的模型参数将约为之前 RNN 的 5 倍，每一轮训练的时间约为 transform_4layer 的三倍，所以我将 max_epoch 设置为了 10，让其能够匹配上预估的时间，此时的 Bleu 为 24.91。如果将 max_epoch 设置为 30，最终的 Bleu 可以达到 27.48。

下面是二者实验对比。

![](https://blogby.oss-cn-guangzhou.aliyuncs.com/20231115174121.png)



## Boss baseline (30.08)

- 应用 back-translation (`TODO`)

  这里我们需要交换实验配置 config 中的 source_lang 和 target_lang，并修改 savedir，训练一个 back-translation 模型后再修改回原来的 config。
  
  然后你需要将 TODO 的部分完善，修改并复用之前的函数就可以达到目的。
  
  （为了与预估时间匹配，这里将 max_epoch 设置为 30 进行实验。）

[代码仓库](https://github.com/Hoper-J/HUNG-YI_LEE_Machine-Learning_Homework)中分享的 Boss 代码展示的是最终训练的结果，完整的运行流程是：

1. 将`实验配置中 ` / `Configuration for experiments` 的 **BACK_TRANSLATION** 设置为 **True** 运行
   训练一个 back-translation 模型，并处理好对应的语料。
2. 将`实验配置` / `Configuration for experiments` 中的 **BACK_TRANSLATION** 设置为 **False** 运行
   结合 ted2020 和 mono (back-translation) 的语料进行训练。

# Gradescope

## Visualize Positional Embedding

你可以直接在 `确定用于生成 submission 的模型权重` / `Confirm model weights used to generate submission` 后进行处理，在仓库的代码中我已经提前注释掉了 `训练循环` / `Training loop` 中的训练部分，如果在之前，模型没有训练，直接运行代码会报错。

![image-20231119122408389](https://blogby.oss-cn-guangzhou.aliyuncs.com/20231119122408.png)

添加的处理代码如下（可以复制下面的处理代码放到你的 submission 模块之后）：

> 推荐阅读：[All Pairs Cosine Similarity in PyTorch](https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572)

```python
pos_emb = model.decoder.embed_positions.weights.cpu().detach()

# 计算余弦相似度矩阵
def get_cosine_similarity_matrix(x):
    x = x / x.norm(dim=1, keepdim=True)
    sim = torch.mm(x, x.t())
    return sim
    
sim = get_cosine_similarity_matrix(pos_emb)
#sim = F.cosine_similarity(pos_emb.unsqueeze(1), pos_emb.unsqueeze(0), dim=2) # 一样的

# 绘制位置向量的余弦相似度矩阵的热力图
plt.imshow(sim, cmap="hot", vmin=0, vmax=1)
plt.colorbar()

plt.show()
```

## Clipping Gradient Norm

只需要将 config.wandb 设置为 True 即可，此时可以在 wandb 上查看。

![image-20231119183413555](https://blogby.oss-cn-guangzhou.aliyuncs.com/20231119183413.png)

或者直接在 train_one_epoch 添加一下处理代码，记录 gnorm。

![image-20231119183535765](https://blogby.oss-cn-guangzhou.aliyuncs.com/20231119183535.png)

