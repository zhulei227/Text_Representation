
该项目主要基于gensim对BOW,TFIDF,LDA,LSI,W2V等传统的文本表示模型进行简单的封装，并添加了chi2,互信息等特征选择方法。
### 一.BOW模型


```python
from nlp_utils import *
import warnings
warnings.filterwarnings("ignore")
```


```python
BOW = BOWModel(where_is_corpus='./corpus/demo.txt',#语料目录
               how_to_extract_text=lambda line: line.split(',')[1],#原始文本的按行提取规则，这里可能会是复杂的正则表达式（据具体情况）
               how_to_segment_word=SegmentWordModel().cut_with_preprocess,#按行的分词，这里基于jieba进行分词，当然也可以使用dnn的方法，自行封装即可
               where_store_dictionary='./dictionary/bow.dict')#bow模型的保存路径
```


```python
BOW.transfer()#构建模型
```

    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\19357\AppData\Local\Temp\jieba.cache
    Loading model cost 0.699 seconds.
    Prefix dict has been built succesfully.
    




    <nlp_utils.BOWModel at 0x2932641ec88>



### 二.特征选择
特征选择模块是对现有的BOW模型进行更新，通过FeatureSelectModel对象进行协助，`how_to_select_feature`可以指定tf（词频）、mi（互信息）、ig（信息增益）、chi2（卡方选择）以及gini（gini指数）进行特征选择，`keep_feature_num`指定需要保留多少特征，`how_to_extract_label`用于指定文本label的读取方式（需要传入一个函数），比如如下选择词频最高的前20个特征。


```python
BOW.update_corpus_model_with_feature_select(feature_select_model=FeatureSelectModel(keep_feature_num=20,
                                                                                    how_to_extract_label=lambda line: int(line.split(',')[0])),
                                            how_to_select_feature='tf')

```

### 三.编码单篇文本/文本集
可以利用`encode_ndarray_text`函数对单篇文本进行编码，利用`encode_ndarray_texts`对文本集进行编码


```python
bow_matrxit = BOW.encode_ndarray_texts(["微信付款", "微信付款可以吗"])
bow_vector = BOW.encode_ndarray_text("微信付款可以么")
print('bow_matrix:\n',bow_matrxit)
print('bow_vector:\n',bow_vector)
```

    bow_matrix:
     [[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
    bow_vector:
     [0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    

后续模型的构建与BOW类似，需要注意的是：  
1）TFIDF的构建需要依赖于现有的BOW，  
2）LDA需要依赖于现有的BOW，  
3）LSI需要依赖于现有的TFIDF
### 四.TFIDF模型


```python
TFIDF = TfidfModel(base_bow_model=BOW, where_store_tfidf_model='./corpus_model/tfidf.model')
TFIDF.transfer()
tfidf_matrix = TFIDF.encode_ndarray_texts(["微信付款", "直接微信转账"])
tfidf_vector = TFIDF.encode_ndarray_text("微信付款可以么微信付款可以么微信付款可以么微信付款可以么微信付款可以么")
print('tfidf_matrix:\n',tfidf_matrix)
print('tfidf_vector:\n',tfidf_vector)
```

    tfidf_matrix:
     [[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    tfidf_vector:
     [0.        0.9825163 0.        0.        0.        0.        0.
     0.        0.1861767 0.        0.        0.        0.        0.
     0.        0.        0.        0.        0.        0.       ]
    

### 五.LDA模型


```python
LDA = LDAModel(base_corpus_model=BOW, where_store_lda_model='./corpus_model/lda.model', topic_num=10)
LDA.transfer()
lda_matrix = LDA.encode_ndarray_texts(["微信付款", "直接微信转账"])
lda_vector = LDA.encode_ndarray_text("微信付款可以么微信付款可以么微信付款可以么微信付款可以么微信付款可以么")
print('lda_matrix:\n',lda_matrix)
print('lda_vector:\n',lda_vector)
```

    lda_matrix:
     [[0.5498947  0.05001274 0.05002429 0.05001787 0.05000004 0.05000698
      0.05000004 0.05000863 0.05001523 0.05001946]
     [0.05002203 0.05001203 0.5499015  0.05001691 0.05000004 0.05000659
      0.05000004 0.05000815 0.05001438 0.05001838]]
    lda_vector:
     [0.9181736 0.        0.        0.        0.        0.        0.
     0.        0.        0.       ]
    

### 六.LSI模型



```python
LSI = LSIModel(base_corpus_model=TFIDF, where_store_lsi_model='./corpus_model/lsi.model', topic_num=2)
LSI.transfer()
lsi_matrix = LSI.encode_ndarray_texts(["微信付款", "直接微信转账"])
lsi_vector = LSI.encode_ndarray_text("微信付款可以么微信付款可以么微信付款可以么微信付款可以么微信付款可以么")
print('lsi_matrix:\n',lsi_matrix)
print('lsi_vector:\n',lsi_vector)
```

    lsi_matrix:
     [[-0.19483276  0.04543959]
     [-0.19483276  0.04543959]]
    lsi_vector:
     [-0.22905156 -0.27707726]
    

### 七.W2V模型
这里W2V只是简单的对文本中所有词的word2vec向量取平均


```python
W2V = Word2VecModel(where_is_corpus='./corpus/demo.txt',
                    how_to_extract_text=lambda line: line.split(',')[1],
                    how_to_segment_word=SegmentWordModel().cut_with_preprocess,
                    where_store_w2v_model='./corpus_model/w2v.model', train_params={'size':2})
W2V.transfer()
w2v_matrix = W2V.encode_ndarray_texts(["微信付款", "直接微信转账"])
w2v_vector = W2V.encode_ndarray_text("微信付款")
print('w2v_matrix:\n',w2v_matrix)
print('w2v_vector:\n',w2v_vector)
```

    w2v_matrix:
     [[ 0.08883715 -0.1966573 ]
     [ 0.08883715 -0.1966573 ]]
    w2v_vector:
     [ 0.08883715 -0.1966573 ]
    

### 八.模型保存


```python
BOW.save_corpus_model()
TFIDF.save_corpus_model()
LDA.save_corpus_model()
LSI.save_corpus_model()
W2V.save_corpus_model()
```
