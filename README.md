<h1>SHE: Sentiment Hashtag Embedding Through Multitask Learning</h1>

<h4><a href="https://sites.google.com/view/gyanendro">Loitongbam Gyanendro Singh</a>, Akash Anil, Sanasam Ranbir Singh</h4>

<p>
Recent studies have shown the importance of utilizing hashtags for sentiment analysis task on social media data. However, as the hashtag generation process is less restrictive, it throws several challenges such as hashtag normalization, topic modeling, semantic similarity, etc. Recently, researchers have tried to address the above challenges through representation learning. However, most of the studies on hashtag embedding try to capture the semantic distribution of hashtags and often fail to capture the sentiment polarity. Further, generating a task-specific hashtag embedding can distort its semantic representation, which is undesirable for sentiment representation of hashtag. Therefore, this paper proposes a semi-supervised Sentiment Hashtag Embedding (SHE) model, which is capable of preserving both
semantic as well as sentiment distribution of the hashtags. In particular, SHE leverages a multitask learning approach using an Autoencoder and a Convolutional Neural Network based classifier. To assess the efficacy on hashtag embedding, we compare the performance of SHE against suitable baselines for three different tasks, namely hashtag sentiment classification, tweet sentiment classification, and retrieval of semantically similar hashtags. It is evident from various experimental results that SHE outperforms the majority of the baselines with significant margins.
</p>


<h4>Proposed framework</h4>
<img src="https://github.com/gloitongbam/SHE/blob/master/SHE.png" alt="Framework">
<br>

## Requirement
The software can run on CPU or GPU, dependency requirements are following:

+ python
+ tensorflow

```shell
pip install numpy
pip install keras
pip install gensim
```

## This work is published in [IEEE TRANSACTIONS ON COMPUTATIONAL SOCIAL SYSTEMS 2020](https://ieeexplore.ieee.org/abstract/document/8963749)


