import os, sys, re, json
import zipfile
import numpy as np
from nltk.tokenize import RegexpTokenizer


workpath = os.path.dirname(os.path.abspath('utils/utils.py'))


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_json_in_zip(filename):
    ''' from zip file extract the json

    filename: files as "*.zip"
    '''
    with zipfile.ZipFile(filename, "r") as archive:
        names = archive.namelist()
        json_file = [json.loads(archive.read(name).decode("utf-8"))\
                     for name in names]
    return json_file


def choose_most_activ_ans(ans_map):
    tmp_dict = {}
    scores = {'yes': 1, 'maybe': 0.5, 'no': 0}
    for ans in ans_map['answers']:
        score = scores[ans['answer_confidence']]
        answer = ans['answer']
        if answer in tmp_dict:
            tmp_dict[answer] += score
        else:
            tmp_dict[answer] = score
    return max(tmp_dict.keys(), key=lambda k: tmp_dict[k])


def search_top_answers(ans_data, numbers=1000):
    '''search top x frequent answers

    '''
    ans_dict = {}
    for ans_map in ans_data:
        activ_ans = choose_most_activ_ans(ans_map)
        if activ_ans not in ans_dict:
            ans_dict[activ_ans] = 1
        else:
            ans_dict[activ_ans] += 1
    sorted_keys = sorted(ans_dict.keys(),
                    key=lambda k: ans_dict[k], reverse=True)
    return sorted_keys[:numbers]


def parse_dataset(ques_file, ans_file, task='Multiple-Choice', train=True):
    ''' parse the coco-qa dataset
        return the tri-tuples whose answer is in top answers
    input:
        ques_file: <str>  question file name
        ans_file : <str>  answer file name
        task     : <str>  the type of question
        train    : <bool> train set(True) or test set(False)

    return:
        results  : <list<tuple>>
            [(image_id, question, answer)]
        ans_one_hot : look up table for answer
    '''
    if task == 'Open-Ended':
        idx = 0
    elif task == 'Multiple-Choice':
        idx = 1
    else:
        raise ValueError("No question type named {}".format(task))
    ques_data = load_json_in_zip(ques_file)[idx]
    ans_data = load_json_in_zip(ans_file)[0]
    if not os.path.exists(workpath + '/data/top_answers.json'):
        print("Creating top_answer.json ...")
        top_ans = search_top_answers(ans_data['annotations'])
        with open(workpath + '/data/top_answers.json', "w") as tmpfile:
            json.dump({"top_answers": top_ans}, tmpfile)
    else:
        with open(workpath + '/data/top_answers.json', "r") as tmpfile:
            top_ans = json.loads(tmpfile.read())["top_answers"]
    answers_map = {}
    for ans_map in ans_data['annotations']:
        activ_ans = choose_most_activ_ans(ans_map)
        if activ_ans not in top_ans and train:
            continue
        answers_map[ans_map['question_id']] = activ_ans
    ans_one_hot = {top_ans[i]:i for i in range(len(top_ans))}
    results = []
    for ques_map in ques_data['questions']:
        if ques_map['question_id'] not in answers_map:
            continue
        answer = answers_map[ques_map['question_id']]
        results.append((ques_map['image_id'], ques_map['question'],
                    answer))
    return results, ans_one_hot


def load_wv_50(vocabfile, wvfile):
    wv_list = np.loadtxt(wvfile, dtype=float)
    with open(vocabfile) as vf:
        words = [line.strip() for line in vf]
    num_to_word = dict(enumerate(words))
    word_to_num = invert_dict(num_to_word)
    return wv_list, word_to_num, num_to_word


def load_glove_wv_300(wvfile):
    cnt = 0
    word_to_num = {}
    wv_list = []
    with open(wvfile, "r") as f:
        for line in f:
            tmp = line.strip('\n').split(' ')
            word = tmp[0]
            wv = np.array([float(val) for val in tmp[1:]])
            wv_list.append(wv)
            if word in word_to_num:
                print(word)
                break
            word_to_num[word] = cnt
            cnt += 1
            sys.stdout.write("\r {} word loaded".format(cnt))
            sys.stdout.flush()
    num_to_word = invert_dict(word_to_num)
    return wv_list, word_to_num, num_to_word


def ques_to_word_list(raw_ques):
    tokenizer = RegexpTokenizer(r'\w+')
    word_list = [token.lower() for token in tokenizer.tokenize(raw_ques)]
    word_list.append('?')
    return word_list

def match_image_file(image_id, namelist):
    for name in namelist:
        if re.search('0'*6 + img_id + '\.', name):
            return name
