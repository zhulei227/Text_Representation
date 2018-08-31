from nlp_utils import *

BOW = BOWModel(where_is_corpus='./corpus/demo.txt',
               how_to_extract_text=lambda line: line.split(',')[1],
               how_to_segment_word=SegmentWordModel().cut_with_preprocess,
               where_store_dictionary='./dictionary/bow.dict')
BOW.transfer()
BOW.update_corpus_model_with_feature_select(feature_select_model=FeatureSelectModel(keep_feature_num=20,
                                                                                    how_to_extract_label=lambda line: int(line.split(',')[0])),
                                            how_to_select_feature='gini')

bow_matrxit = BOW.encode_ndarray_texts(["微信付款", "微信付款可以吗"])
bow_vector = BOW.encode_ndarray_text("微信付款可以么")
print('bow_matrix:\n',bow_matrxit)
print('bow_vector:\n',bow_vector)


TFIDF = TfidfModel(base_bow_model=BOW, where_store_tfidf_model='./corpus_model/tfidf.model')
TFIDF.transfer()
tfidf_matrix = TFIDF.encode_ndarray_texts(["微信付款", "直接微信转账"])
tfidf_vector = TFIDF.encode_ndarray_text("微信付款可以么微信付款可以么微信付款可以么微信付款可以么微信付款可以么")
print('tfidf_matrix:\n',tfidf_matrix)
print('tfidf_vector:\n',tfidf_vector)

LDA = LDAModel(base_corpus_model=TFIDF, where_store_lda_model='./corpus_model/lda.model', topic_num=2)
LDA.transfer()
lda_matrix = LDA.encode_ndarray_texts(["微信付款", "直接微信转账"])
lda_vector = LDA.encode_ndarray_text("微信付款可以么微信付款可以么微信付款可以么微信付款可以么微信付款可以么")
print('lda_matrix:\n',lda_matrix)
print('lda_vector:\n',lda_vector)


LSI = LSIModel(base_corpus_model=TFIDF, where_store_lsi_model='./corpus_model/lsi.model', topic_num=2)
LSI.transfer()
lsi_matrix = LSI.encode_ndarray_texts(["微信付款", "直接微信转账"])
lsi_vector = LSI.encode_ndarray_text("微信付款可以么微信付款可以么微信付款可以么微信付款可以么微信付款可以么")
print('lsi_matrix:\n',lsi_matrix)
print('lsi_vector:\n',lsi_vector)
W2V = Word2VecModel(where_is_corpus='./corpus/demo.txt',
                    how_to_extract_text=lambda line: line.split(',')[1],
                    how_to_segment_word=SegmentWordModel().cut_with_preprocess,
                    where_store_w2v_model='./corpus_model/w2v.model', train_params={'size':2})
W2V.transfer()
w2v_matrix = W2V.encode_ndarray_texts(["微信付款", "直接微信转账"])
w2v_vector = W2V.encode_ndarray_text("微信付款")
print('w2v_matrix:\n',w2v_matrix)
print('w2v_vector:\n',w2v_vector)

BOW.save_corpus_model()
TFIDF.save_corpus_model()
LDA.save_corpus_model()
LSI.save_corpus_model()
W2V.save_corpus_model()