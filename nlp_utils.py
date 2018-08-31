import jieba
from jieba.analyse import extract_tags
import os
import re
import gensim
from gensim import corpora,models
from gensim.models import LdaModel,LsiModel,Word2Vec
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.feature_selection import chi2,SelectKBest,mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

'''
分词模块
'''
class SegmentWordModel(object):
    def __init__(self,user_dicts_directory=None):
        '''
        :param user_dicts_directory:自定义词典文件夹
        '''
        # 尝试并行分词(unix系统可以，windows不可以)
        try:
            from multiprocessing import cpu_count
            jieba.enable_parallel(cpu_count() - 1)
        except:
            pass
        # 加载自定义词典
        if user_dicts_directory is not None:
            filenames = os.listdir(user_dicts_directory)
            for filename in filenames:
                jieba.load_userdict(user_dicts_directory + filename)

    def cut_for_all(self, text):
        seg_list = jieba.cut(text, cut_all=True)
        return ' '.join(seg_list)

    def cut_for_simple(self, text):
        seg_list = jieba.cut(text)
        return ' '.join(seg_list)

    def cut_for_search(self, text):
        seg_list = jieba.cut_for_search(text)
        return ' '.join(seg_list)

    def cut_for_all_and_search(self, text):
        seg_set = set()
        for item in jieba.cut_for_search(text):
            seg_set.add(item)
        for item in jieba.cut(text, cut_all=True):
            seg_set.add(item)
        return ' '.join(seg_set)

    def cut_with_preprocess(self,text):
        r = '[^0-9a-zA-Z\u4E00-\u9FA5]'  # 只保留数字、字母以及汉字
        text = re.sub(r, '', text)
        text = re.sub('[0-9]+\.?[0-9]*', 'NUM', text)  # 替换所有数字为字符NUM
        text = re.sub('[a-zA-Z0-9]{10,}', '', text)  # 去掉长度超过10的英文字符
        return ' '.join(jieba.cut(text))

    def extract_keywords(self, text, withWeight=False, topK=10):
        tags_list = extract_tags(text, withWeight=withWeight, topK=topK)  # 默认是tfidf值
        return tags_list
'''
文本特征选择模块,包括:
无监督特征选择:
1.词频
2.idf
有监督特征选择:
1.ig
2.gini
3.chi2
4.mi
'''
class FeatureSelectModel(object):

    def __init__(self,keep_feature_num=1000,where_is_corpus=None,how_to_extract_text=None,how_to_extract_label=None):
        """
        :param keep_feature_num: 需要保留的特征数(Int)
        :param where_is_corpus:语料位置
        :param how_to_extract_text:如何抽取语料中的文本数据
        :param how_to_extract_label:如何抽取类别标签,监督式的特征选择需要用到(Function)
        """
        self.keep_feature_num=keep_feature_num
        self.corpus_path=where_is_corpus
        self.extract_text_operator=how_to_extract_text
        self.extract_label_operator=how_to_extract_label

    def set_bow_model(self,bow_model):
        """
        利用bow_model 抽取需要训练的数据

        :param corpus_model:
        :return:
        """
        self.bow_model=bow_model
        self.corpus_path=bow_model.corpus_path if self.corpus_path is None else self.corpus_path
        if self.corpus_path is None:
            raise ValueError('please set FeatureSelectModel\'s where_is_corpus param or set_bow_model() with where_is_corpus param')

        corpus_texts = []
        corpus_labels = []
        for line in open(self.corpus_path, encoding='utf8'):
            try:
                extract_text = bow_model.extract_text_operator(line)
                extract_label = None
                if self.extract_label_operator:
                    extract_label = self.extract_label_operator(line)
                corpus_texts.append(extract_text)
                if extract_label is not None:
                    corpus_labels.append(extract_label)
            except:
                continue
        self.train_x = bow_model.encode_ndarray_texts(texts=corpus_texts)
        if len(corpus_labels) != 0:
            self.train_y = np.asarray(corpus_labels)

        return self
    def update_bow_model_by_feature_select(self,how_to_select_feature='gini'):
        if how_to_select_feature=='gini':
            self.update_bow_model_by_gini_select()
        elif how_to_select_feature=='chi2':
            self.update_bow_model_by_chi2_select()
        elif how_to_select_feature=='ig':
            self.update_bow_model_by_ig_select()
        elif how_to_select_feature=='mi':
            self.update_bow_model_by_mi_select()
        elif how_to_select_feature=='tf':
            self.update_bow_model_by_tf_select()
        else:
            raise ValueError('no implement the feature select method: '+how_to_select_feature)
    def _update_bow_model(self,weights):
        """
        更新bow_model中的dictionary
        :param weights: dictionary中对应的weight
        :return:
        """
        if len(weights) > self.keep_feature_num:
            weight_indexs = zip(weights, range(len(weights)))
            sorted_weight_indexs = sorted(weight_indexs, reverse=True)
            sorted_indexs = [x[1] for x in sorted_weight_indexs]

            filter_indexs = sorted_indexs[self.keep_feature_num:]#需要过滤掉的index
            self.bow_model.dictionary.filter_tokens(filter_indexs)
            self.bow_model.dictionary.compactify()
            self.bow_model.save_corpus_model()
    def update_bow_model_by_chi2_select(self):
        fit_info = (SelectKBest(score_func=chi2, k='all').fit(self.train_x, self.train_y))
        weights=fit_info.scores_.tolist()
        self._update_bow_model(weights)
    def update_bow_model_by_mi_select(self):
        fit_info = (SelectKBest(score_func=mutual_info_classif, k='all').fit(self.train_x, self.train_y))
        weights = fit_info.scores_.tolist()
        self._update_bow_model(weights)
    def update_bow_model_by_tf_select(self):
        '''
        根据词频做特征选择
        :return:
        '''
        weights=np.sum(self.train_x,axis=0).tolist()
        self._update_bow_model(weights)
    def update_bow_model_by_gini_select(self):
        estimator = RandomForestClassifier(criterion='gini')
        estimator.fit(self.train_x, self.train_y)
        weights = estimator.feature_importances_.tolist()
        self._update_bow_model(weights)
    def update_bow_model_by_ig_select(self):
        estimator = RandomForestClassifier(criterion='entropy')
        estimator.fit(self.train_x, self.train_y)
        weights = estimator.feature_importances_.tolist()
        self._update_bow_model(weights)
'''
语料处理模块,bow,tfidf,lsi,lda...
包含基本功能:
1.加载语料
2.训练语料
3.文本编码
'''
class CorpusModel(object):
    """
    语料处理模块,bow,tfidf,lsi,lda,word embeddings...的父类

    包含基本功能的接口方法:
    1.加载语料
    2.训练语料
    3.文本编码
    """
    __metaclass__ = ABCMeta

    def __init__(self,where_is_corpus=None,how_to_extract_text=None,how_to_segment_word=None,where_store_dictionary=None):
        """
        :param where_is_corpus: 语料位置,仅仅训练阶段需要(str)
        :param how_to_extract_text: 从语料中抽取文本的操作,同样仅仅训练阶段需要使用(function)
        :param how_to_segment_word: 分词操作(function)
        :param where_store_dictionary: 字典位置(str)
        """
        self.corpus_path = where_is_corpus
        self.extract_text_operator=how_to_extract_text
        self.seg_operator = how_to_segment_word
        self.dictionary_path=where_store_dictionary

    @abstractmethod
    def load_corpora_documents(self):
        """
        需要指定corpus路径、文本抽取方法等信息

        加载训练语料库,仅仅训练阶段需要使用
        :return:
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def extract_corpus_data(self):
        """
        需要指定corpus路径、文本抽取方法等信息

        提取corpus类封装的训练数据,方便其他模型的进一步训练
        :return:
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def extract_ndarray_data(self):
        """
        需要指定corpus路径、文本抽取方法等信息

        提取ndarray类封装的训练数据,方便其他模型的进一步训练
        :return:
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def save_corpus_model(self):
        """
        保存语料训练后的模型
        :return:
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def load_corpus_model(self):
        """
        加载语料训练模型
        :return:
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def encode_corpus_texts(self,texts):
        """
        编码文本集(corpus)

        :param texts: 需要编码的文本集(Iterator[String])
        :return: corpus
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def encode_ndarray_texts(self,texts):
        """
        编码文本集

        :param texts: 需要编码的文本集(Iterator[String])
        :return: ndarray
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def encode_corpus_text(self,text):
        """
        编码文本(corpus格式)

        :param text:需要编码的单个文本(String)
        :return: corpus
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def encode_ndarray_text(self,text):
        """
        编码文本(ndarray格式)

        :param text: 需要编码的单个文本(String)
        :return: ndarray
        """
        raise RuntimeError("need to implement!")
    @abstractmethod
    def transfer(self):
        """
        训练语料模型
        :return:
        """
        raise RuntimeError("need to implement!")
class TopicModel(CorpusModel):
    def __init__(self,base_corpus_model,where_store_topic_model,topic_num,topic_model_class):
        """
        主题模型类,lda,lsi的父类

        :param base_corpus_model: 基本的corpus_model，一般为BOW或者TFIDF
        :param where_store_topic_model: 存储主题模型的路径
        :param topic_num: 主题数
        :param topic_model_class: 主题类,比如gensim.models.LsiModel
        """
        CorpusModel.__init__(self)
        self.base_corpus_model=base_corpus_model
        self.topic_model_path=where_store_topic_model
        self.topic_num=topic_num
        self.topic_model_class = topic_model_class


    def load_corpora_documents(self):
        return self.base_corpus_model.load_corpora_documents()

    def extract_corpus_data(self):
        return self.topic_model[self.base_corpus_model.extract_corpus_data()]

    def extract_ndarray_data(self):
        return gensim.matutils.corpus2dense(self.extract_corpus_data(), num_terms=self.topic_num)

    def save_corpus_model(self):
        self.topic_model.save(self.topic_model_path)

    def load_corpus_model(self):
        try:
            self.topic_model = self.topic_model_class.load(self.topic_model_path)
        except:
            self.transfer()

    def encode_corpus_texts(self, texts):
        return self.topic_model[self.base_corpus_model.encode_corpus_texts(texts)]

    def encode_corpus_text(self, text):
        return self.encode_corpus_texts(texts=[text])[0]

    def encode_ndarray_texts(self, texts):
        return gensim.matutils.corpus2dense(self.encode_corpus_texts(texts), num_terms=self.topic_num).transpose()

    def encode_ndarray_text(self, text):
        return self.encode_ndarray_texts(texts=[text])[0]

    def transfer(self, other_texts=None):
        """
        利用base_corpus_model对自带/其它语料进行训练

        :param other_texts: Iterator[String]
        :return:
        """
        if other_texts is not None:
            self.topic_model = self.topic_model_class(corpus=self.base_corpus_model.encode_corpus_texts(texts=other_texts),num_topics=self.topic_num)
        else:
            self.topic_model = self.topic_model_class(corpus=self.base_corpus_model.extract_corpus_data(),num_topics=self.topic_num)
        return self
class BOWModel(CorpusModel):
    def __init__(self,where_is_corpus,how_to_extract_text,how_to_segment_word,where_store_dictionary):
        """
        初始化对象阶段会加载模型,如果没有则训练
        :param where_is_corpus:
        :param how_to_extract_text:
        :param how_to_segment_word:
        :param where_store_dictionary:
        """
        CorpusModel.__init__(self,where_is_corpus,how_to_extract_text,how_to_segment_word,where_store_dictionary)
    def load_corpora_documents(self):
        if self.corpus_path is None:
            raise RuntimeError("bow model\'s corpus path can not be None,you should set bow model\'s param,like:bow_model.where_is_corpus='./courpus.txt'")
        corpora_documents = []
        for line in open(self.corpus_path, encoding='utf8'):
            text = self.extract_text_operator(line)
            corpora_documents.append(self.seg_operator(text).split(' '))
        return corpora_documents
    def extract_corpus_data(self):
        return [self.dictionary.doc2bow(word_list) for word_list in self.load_corpora_documents()]
    def extract_ndarray_data(self):
        return gensim.matutils.corpus2dense(self.extract_corpus_data(), num_terms=len(self.dictionary))
    def save_corpus_model(self):
        self.dictionary.save(self.dictionary_path)
    def load_corpus_model(self):
        try:
            self.dictionary=corpora.Dictionary.load(self.dictionary_path)
        except:
            self.transfer()
    def update_corpus_model_with_feature_select(self,feature_select_model=None,how_to_select_feature='gini'):
        '''
        根据特征选择方法更新dictionary
        :param feature_select_model: 特征选择模型
        :param how_to_select_feature: 特征选择方式
        :return:
        '''
        feature_select_model.set_bow_model(self).update_bow_model_by_feature_select(how_to_select_feature=how_to_select_feature)
    def encode_corpus_texts(self,texts):
        return [self.dictionary.doc2bow(self.seg_operator(text).split(' ')) for text in texts]
    def encode_corpus_text(self,text):
        return self.encode_corpus_texts(texts=[text])[0]
    def encode_ndarray_texts(self,texts):
        return gensim.matutils.corpus2dense(self.encode_corpus_texts(texts), num_terms=len(self.dictionary)).transpose()
    def encode_ndarray_text(self,text):
        return self.encode_ndarray_texts(texts=[text])[0]
    def transfer(self):
        corpora_documents = self.load_corpora_documents()
        self.dictionary = corpora.Dictionary(corpora_documents)
        return self
class TfidfModel(CorpusModel):
    def __init__(self,base_bow_model,where_store_tfidf_model):
        """
        :param base_bow_model: 训练好的bow模型
        :param where_store_tfidf_model: tfidf模型存储路径
        """
        CorpusModel.__init__(self)
        self.tfidf_model_path = where_store_tfidf_model
        #部分功能基于bow模型来实现
        self.bow_model=base_bow_model

    def load_corpora_documents(self):
        return self.bow_model.load_corpora_documents()
    def extract_corpus_data(self):
        return self.tfidf_model[self.bow_model.extract_corpus_data()]
    def extract_ndarray_data(self):
        return gensim.matutils.corpus2dense(self.extract_corpus_data(), num_terms=len(self.bow_model.dictionary))
    def save_corpus_model(self):
        self.tfidf_model.save(self.tfidf_model_path)
    def load_corpus_model(self):
        try:
            self.tfidf_model=models.TfidfModel.load(self.tfidf_model_path)
        except:
            self.transfer()
    def encode_corpus_texts(self,texts):
        return self.tfidf_model[self.bow_model.encode_corpus_texts(texts)]
    def encode_corpus_text(self,text):
        return self.encode_corpus_texts(texts=[text])[0]
    def encode_ndarray_texts(self,texts):
        return gensim.matutils.corpus2dense(self.encode_corpus_texts(texts), num_terms=len(self.bow_model.dictionary)).transpose()
    def encode_ndarray_text(self,text):
        return self.encode_ndarray_texts(texts=[text])[0]
    def transfer(self):
        self.tfidf_model = models.TfidfModel(self.bow_model.extract_corpus_data())
        return self
class LDAModel(TopicModel):
    def __init__(self,base_corpus_model,where_store_lda_model,topic_num):
        """
        :param base_corpus_model: lda主题模型一般基于bow模型的基础上进行构建
        :param where_store_lda_model:存储lda模型位置
        :param topic_num:主题数
        """
        TopicModel.__init__(self, base_corpus_model=base_corpus_model,
                            where_store_topic_model=where_store_lda_model, topic_num=topic_num,
                            topic_model_class=LdaModel)
class LSIModel(TopicModel):
    def __init__(self,base_corpus_model,where_store_lsi_model,topic_num):
        """
        :param base_corpus_model: lsi主题模型一般基于tfidf模型的基础上进行构建
        :param where_store_lsi_model:存储lsi模型位置
        :param topic_num:主题数
        """
        TopicModel.__init__(self,base_corpus_model=base_corpus_model,where_store_topic_model=where_store_lsi_model,topic_num=topic_num,topic_model_class=LsiModel)
class Word2VecModel(CorpusModel):
    def __init__(self,where_is_corpus=None,how_to_extract_text=None,how_to_segment_word=None,where_store_w2v_model=None,train_params=None):
        """
        训练word2vec词向量,并利用向量的平均值对文本进行编码

        :param where_is_corpus: corpus路径(String)
        :param how_to_extract_text: 如何抽取文本语料(function)
        :param how_to_segment_word: 如何分词(function)
        :param where_store_w2v_model: 存储word2vec路径(String)
        :param train_params: 训练参数
        """
        CorpusModel.__init__(self,where_is_corpus=where_is_corpus,how_to_extract_text=how_to_extract_text,how_to_segment_word=how_to_segment_word,where_store_dictionary=None)
        self.w2v_model_path=where_store_w2v_model
        self.train_params={} if train_params is None else train_params
    def load_corpora_documents(self):
        if self.corpus_path is None:
            raise RuntimeError("model\'s corpus path can not be None,you should set model\'s param,like:model.where_is_corpus='./courpus.txt'")
        corpora_documents = []
        for line in open(self.corpus_path, encoding='utf8'):
            text = self.extract_text_operator(line)
            corpora_documents.append(self.seg_operator(text).split(' '))
        return corpora_documents
    def extract_corpus_data(self):raise RuntimeError('need not to implement!')
    def extract_ndarray_data(self):raise RuntimeError('need not to implement!')
    def save_corpus_model(self):
        self.w2v_model.save(self.w2v_model_path)
    def load_corpus_model(self,other_w2v_model_path=None):
        if other_w2v_model_path:
            pass
        else:
            try:
                self.w2v_model=Word2Vec.load(self.w2v_model_path)
            except:
                self.train()
    def encode_corpus_texts(self,texts):raise RuntimeError('need not to implement!')
    def encode_corpus_text(self,text):raise RuntimeError('need not to implement!')
    def get_word_vector(self,word):
        """
        获取词向量
        :param word:
        :return:
        """
        try:
            vec = self.w2v_model.wv[word]
            return vec, 1
        except:
            return np.zeros(shape=self.w2v_model.vector_size), 0
    def encode_ndarray_texts(self,texts):
        return np.asarray([self.encode_ndarray_text(text).tolist() for text in texts])
    def encode_ndarray_text(self,text):
        """
        获取句子向量,取词向量的平均值
        :param text:
        :return:
        """
        text_vector=np.zeros(shape=self.w2v_model.vector_size)
        seg_txt=self.seg_operator(text)
        if seg_txt:
            count=0
            for word in seg_txt.split(' '):
                word_vector,count_=self.get_word_vector(word)
                text_vector+=word_vector
                count+=count_
            text_vector=text_vector/max(count,1)
        return text_vector
    def transfer(self):
        # default_params={'size':100, 'alpha':0.025, 'window':5, 'min_count':5,
        #          'max_vocab_size':None, 'sample':1e-3, 'seed':1, 'workers':3, 'min_alpha':0.0001,
        #          'sg':0, 'hs':0, 'negative':5, 'cbow_mean':1, 'hashfxn':hash, 'iter':5, 'null_word':0,
        #          'trim_rule':None, 'sorted_vocab':1, 'batch_words':1000, 'compute_loss':False}
        self.w2v_model=Word2Vec(self.load_corpora_documents(),**self.train_params)
        return self
