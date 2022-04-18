from transformers import BertTokenizer, AlbertModel
from transformers import BertConfig
from transformers import BertModel, BertPreTrainedModel
import numpy as np
import torch

model_name = 'service_bert_base_chinese'
MODEL_PATH = 'service_bert_base_chinese/'

tokenizer = BertTokenizer.from_pretrained(model_name)
service_model_config = BertConfig.from_pretrained(model_name)
service_bert_model = BertModel.from_pretrained(MODEL_PATH, config=service_model_config)

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1);
    ssB = (B_mB ** 2).sum(1);
    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def count_corr(text1, text2):
    text_encode1 = tokenizer.encode(text1)
    text_encode2 = tokenizer.encode(text2)
    # print(text_encode)
    text_encode1 = torch.Tensor([text_encode1]).long()
    text_encode2 = torch.Tensor([text_encode2]).long()
    with torch.no_grad():
        outputs1 = service_bert_model(text_encode1)
        outputs2 = service_bert_model(text_encode2)
    n11 = outputs1[0].numpy()
    n22 = outputs2[0].numpy()
    ab = np.array([n11[0][0], n22[0][0]])
    # print(np.corrcoef(ab))
    n1 = np.array(outputs1[0][0])
    n2 = np.array(outputs2[0][0])
    corr = corr2_coeff(n1, n2)
    # print(text1)
    # print(text2)
    # print(corr[0])
    # print(np.mean(corr))
    return corr[0][0]
