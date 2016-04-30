from gensim.models import word2vec

if __name__ == '__main__':
    print('Model loading')
    model = word2vec.Word2Vec.load("question__word_features")
    print('Done')
    word_vector = model.syn0
    word_index = model.index2word
    
    file = open("word-vec.txt", "w")
    for i in range(len(word_index)):
        file.write('#' + word_index[i] + '# ')
        for item in word_vector[i]:
            file.write(str(item) + ' ')
        file.write("\n")
        
    file.close()    