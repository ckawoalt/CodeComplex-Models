# Deep-Learning-based-Code-Complexity-Prediction

# Introduction

Deciding the computational complexity of algorithms is a really challenging problem even for human algorithm experts. Theoretically, the problem of deciding the computational complexity of a given program is undecidable due to the famous Halting problem. In this paper, we tackle the problem by designing a neural network that comprehends the algorithmic nature of codes and estimates the worst-case complexity.

First, we construct a code dataset called the CodeComplex that consists of 4,517 Java codes submitted to programming competitions by human programmers and their complexity labels annotated by a group of algorithm experts. As far as we are aware, the CodeComplex dataset is by far the largest code dataset for the complexity prediction problem. Then, we present several baseline algorithms using the previous code understanding neural models such as CodeBERT, GraphCodeBERT, PLBART, and CodeT5. As the previous code understanding models do not work well on longer codes due to the code length limit, we propose the hierarchical Transformer architecture which takes method-level code snippets instead of whole codes and combines the method-level embeddings to the class-level embedding and ultimately to the code-level embedding. Moreover, we introduce pre-training objectives for the proposed model to induce the model to learn both the intrinsic property of the method-level codes and the relationship between the components.

Lastly, we demonstrate that the proposed hierarchical architecture and pre-training objectives achieve state-of-the-art performance in terms of complexity prediction accuracy compared to the previous code understanding models.

![model_structure](./images/model_structure.png)

# Data

raw data : 
pretrain and finetuned model: http://gofile.me/6UFJt/7U4B75pq1
training and validation data: http://gofile.me/6UFJt/LYjsMGGf0

# 