**语料:**

训练语料来源于[复旦大学中文文本分类语料库](http://www.nlpir.org/download/tc-corpus-answer.rar)

**文本表达模型:**

目前主要基于gensim封装了bow,tfidf,lsi,lda,word_embedding_avg等;基于sklearn的api对bow模型进行特征选择,包括ig,gini,chi2,mi等