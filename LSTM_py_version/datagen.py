from utils import utils
import numpy as np

dataset_path = "/media/scarlet/Virtual/Share/VQA_dataset/coco-qa"

def word_embed(voc, _type="simple"):
    print("Loading pre-trained word vectors")
    if _type == "simple":
        # simple 50-D word vectors
        wv_list, w2n, _ = utils.load_wv_50('wordvectors/vocab.txt',
                               'wordvectors/wordVectors.txt')
    elif _type == "glove.6B.300d":
        # glove 6B.300-D word vectors
        wv_list, w2n, _= utils.load_glove_wv_300('wordvectors/glove.6B.300d.txt')
    else:
        raise ValueError("No such word embedding".format(_type))
    print("Done")
    print("Extract the sub word vectors list")
    sub_wv_list = []
    cnt = 0
    for word in voc.keys():
        if word in w2n:
            sub_wv_list.append(wv_list[w2n[word]])
        else:
            print("Cannot embed '{}', set default".format(word))
            sub_wv_list.append(np.zeros(wv_list[0].shape))
        cnt += 1
    sub_wv_list[len(voc) - 1] = np.zeros(wv_list[0].shape)
    sub_n2w = utils.invert_dict(voc)
    print("Done")
    return np.array(sub_wv_list), voc, sub_n2w


def build_voc_and_get_data(max_ques_len=30):
    print("Read data from disk...")
    qf = dataset_path + "/Questions_Train_mscoco.zip"
    af = dataset_path + "/Annotations_Train_mscoco.zip"
    train_data, ans_lk_tb = utils.parse_dataset(qf, af, 'Open-Ended')

    qf = dataset_path +  "/Questions_Val_mscoco.zip"
    af = dataset_path +  "/Annotations_Val_mscoco.zip"
    test_data, _ = utils.parse_dataset(qf, af, 'Open-Ended', train=False)
    print("Done")
    # Build vocabulary
    print("Building vocabulary...")
    voc = set({})
    for res in train_data:
        question = res[1]
        qword_list = set(utils.ques_to_word_list(question))
        voc |= qword_list
    for res in test_data:
        question = res[1]
        qword_list = set(utils.ques_to_word_list(question))
        voc |= qword_list
    voc = list(voc)
    voc.append("<unk>")
    voc = {voc[i]:i for i in range(len(voc))}
    print("Done")
    # Encoding the question and answer
    print("Encoding...")
    res_train_data = []
    for img_id, ques, ans in train_data:
        ques_encode_list = [voc[word] for word in utils.ques_to_word_list(ques)]
        if len(ques_encode_list) > max_ques_len:
            raise ValueError("length of '{}' of length {} exceeds the max length".format(ques, len(ques_encode_list)))
        ques_encode_list.extend([len(voc) - 1] * (max_ques_len - len(ques_encode_list)))
        res_train_data.append(
            [img_id, ques_encode_list, ans_lk_tb[ans]])
    res_test_data = []
    for img_id, ques, ans in test_data:
        ques_encode_list = [voc[word] for word in utils.ques_to_word_list(ques)]
        if len(ques_encode_list) > max_ques_len:
            raise ValueError("length of '{}' of length {} exceeds the max length".format(ques, len(ques_encode_list)))
        ques_encode_list.extend([len(voc) - 1] * (max_ques_len - len(ques_encode_list)))
        res_test_data.append(
            [img_id, ques_encode_list,
             ans_lk_tb[ans] if ans in ans_lk_tb else -1])
    print("Done")
    return res_train_data, res_test_data, voc, ans_lk_tb


def data_iter(data, batch_size=64):
    data_len = len(data)
    # total_steps = int(np.ceil(len(data) / float(batch_size)))
    total_steps = len(data) // batch_size
    for step in range(total_steps):
        batch_st = step * batch_size
        ques_x = np.array(
            [tmp[1] for tmp in data[batch_st: batch_st + batch_size]])
        y = np.array(
            [tmp[2] for tmp in data[batch_st: batch_st + batch_size]])
        yield (ques_x, y)
