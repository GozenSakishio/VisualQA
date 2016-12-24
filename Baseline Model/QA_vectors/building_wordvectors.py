import logging
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import word2vec
import json


def doc_towordlist(doc, RemoveStopWords = False):
    # Convert all letter to lower case
    doc = doc.lower()
    # Remove non-letter and split it
    tokenizer = RegexpTokenizer(r'\w+')
    wordlist = tokenizer.tokenize(doc)
    # Remove the stop word if necessary
    if RemoveStopWords == True:
        stopw = stopwords.words('english')
        tmpwords = wordlist[:]
        for item in tmpwords:
            if item.lower() in stopw:
                wordlist.remove(item)
    return wordlist


def trainword2vec(doclist):
    # Get a list of list of word
    sentences = []
    for item in doclist:
        sentences += [doc_towordlist(item['question'])]
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Initialize the parameters
    num_features = 300
    min_word_count = 10
    num_workers = 4
    context = 10
    downsampling = 1e-3
    print('Training...')
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
    model.init_sims(replace=True)
    model_name = "question__word_features"
    print('Saving...')
    model.save(model_name)
    print('Done')
    # print(model.most_similar("queen"))
    return model

def outans(ques_id):
    f = open('F:\浙大\科研\SRTP\dataset\mscoco_train2014_annotations.json', 'r')
    jsontext = json.loads(f.readline())
    alist = jsontext['annotations']
    img_id = []
    ques_id = []
    ans_id = []
    ans = []
    ans_conf = []
    i = 0
    for img_ans in alist:
        tmp_imgid = img_ans['image_id']
        tmp_quesid = img_ans['question_id']
        for item in img_ans['answers'] and tmp_quesid in ques_id:
                img_id += [tmp_imgid]
                ques_id += [tmp_quesid]
                ans_id += [item['answer_id']]
                ans += [item['answer'].replace(',', ' ')]
                ans_conf +=  [item['answer_confidence']]
        i += 1
        print('Answer', i, 'processed')  
    print('Write into csv files...')
    output = pd.DataFrame(data={"img_id":img_id, "ques_id":ques_id, "ans_id":ans_id, "answers": ans, "confident": ans_conf})
    output.to_csv( "answers_sample.csv", index=False, quoting=3, quotechar='' )  

if __name__ == '__main__': 
#==============================================================================
     # Read data from .csv files
     print('Reading Trainset for word2vec')
     f = open('F:\浙大\科研\SRTP\dataset\OpenEnded_mscoco_val2014_questions.json', 'r')
     jsontext = json.loads(f.readline())
     # qlist = jsontext['questions']
     qlist = [{'image_id': 0, 'question':'What\'s the weather like today', 'question_id':0}]
     # Word Vectors loading
     try:
         print('Model loading')
         model = word2vec.Word2Vec.load("question__word_features")
         print('Done')
     except:
         print('preModel not found, now get a new trainning model')
         model = trainword2vec(qlist)
     
     # Add vectors
     word_vector = model.syn0
     word_index = model.index2word
     img_id = []
     ques_id = []
     ques_vector = []
     i = 0
     for item in qlist:
         tmpques = doc_towordlist(item['question'])
         tmp_vector = np.zeros(word_vector.shape[1])
         for word in tmpques:
             if word in word_index:
                 tmp_vector += word_vector[word_index.index(word)]
         tmp_imgid = np.zeros(word_vector.shape[1]).astype(np.int32) + item['image_id']
         tmp_quesid = np.zeros(word_vector.shape[1]).astype(np.int32) + item['question_id']
         # 300 rows per question
         ques_vector += tmp_vector.tolist()
         img_id += tmp_imgid.tolist()
         ques_id += tmp_quesid.tolist() 
         i += 1
         if i > 99:
             break;
         print('Question', i, 'processed')
     print('Write into csv files...')
     output = pd.DataFrame(data={"img_id":img_id, "ques_id":ques_id, "vectors":ques_vector})
     output.to_csv( "que_sample.csv", index=False, quoting=3 )
#==============================================================================
     outans(ques_id)
