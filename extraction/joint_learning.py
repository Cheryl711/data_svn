from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import extraction.tf_metrics
import extraction.modeling
import extraction.optimization_finetuning as optimization
import extraction.tokenization
import tensorflow as tf
import re
import time
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
import json
import pickle


import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

global maintext
maintext=''

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "./data/",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    # "bert_config_file", "extraction/checkpoint/albert_config_tiny.json",
    "bert_config_file", "./checkpoint/albert_config_tiny.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "NER", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "./checkpoint/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "output_dir", "./Model_j/",
    "The output directory where the model checkpoints will be written.")

## NOT USED AT ALL
flags.DEFINE_integer(
    "lstm_size", 256,
    "LSTM hidden cell number."
)
flags.DEFINE_string(
    "cell", "lstm",
    "lstm or gru dicnn will be add in feature."
)
flags.DEFINE_integer(
    "num_layers", 1,
    "rnn layers number"
)


## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 50.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification"""

    def __init__(self, guid, text_a, text_b=None, label_ea=None, label_eb=None, label_r=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label_e: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
            label_r: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_ea = label_ea
        self.label_eb = label_eb
        self.label_r = label_r


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, labele_ids, labelr_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labele_ids = labele_ids
        self.labelr_id = labelr_id

def _decode_sentence(text):
    """Convert labeled sentence to (sentence,label) pair"""
    t_str = re.sub(r"{{(\w+):([^}]*)}}",r"{{&$\1:\2$&}}",text)
    tokens = [x for x in re.split("{{|}}", t_str) if len(x) > 0]
    word_list, label_list = [], []
    for token in tokens:
        if token[0:2] == '&$':
            token = token[2:-2]
            tag, token = token.split(':')[0], token.split(":")[1]
            for i, c in enumerate(token):
                word_list.append(c)
                if i == 0:
                    label_list.append('B-'+tag)
                else:
                    label_list.append('I-'+tag)
        else:
            for c in token:
                word_list.append(c)
                label_list.append('O')
    return ' '.join(word_list), ' '.join(label_list)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labeles(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_labelrs(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, test=False):
        """Read data.

        Example:
            label_r, label_r, seq_1, seq_2, seq_3, timestamp
            label_r, seq_1, timestamp
        """
        df = open(input_file, 'r', encoding='utf8')
        # df = input_file.read().decode()
        # a = df.splitlines()
        lines = []
        if test:
            for line in df:
                print(line)
                example_dict = {}
                line = re.sub('\s+', '', line)
                line_fields = line.strip().split(',')[:-1]
                line_fields_num = len(line_fields)
                if line_fields_num == 1:
                    text, label_e = _decode_sentence(line_fields[0])
                    example_dict["text_a"] = text
                    example_dict["text_b"] = text
                    example_dict["label_ea"] = label_e
                    example_dict["label_eb"] = label_e
                    example_dict["label_r"] = "single"
                    lines.append(example_dict)
                    example_dict = {}
                else:
                    for text_a, text_b in zip(line_fields[:-1], line_fields[1:]):
                        example_dict["text_a"], example_dict["label_ea"] = _decode_sentence(text_a)
                        example_dict["text_b"], example_dict["label_eb"] = _decode_sentence(text_b)
                        example_dict["label_r"] = "single"
                        lines.append(example_dict)
                        example_dict = {}
            # print(lines)
            return lines

        for line in df:
            example_dict = {}
            line = re.sub('\s+', '', line)
            line_fields = line.strip().split(',')[:-1]  ## split and remove timestamp
            line_fields_num = len(line_fields)
            if line_fields_num < 2:
                tf.logging.info(" Bad line: %s", line)
                continue
            if line_fields_num % 2 == 0 and line_fields_num != 2:
                tf.logging.info(" Bad line: %s", line)
                continue
            if line_fields_num % 2 == 0:
                text, label_e = _decode_sentence(line_fields[1])
                example_dict["text_a"] = text
                example_dict["text_b"] = text
                example_dict["label_ea"] = label_e
                example_dict["label_eb"] = label_e
                example_dict["label_r"] = line_fields[0]
                lines.append(example_dict)
                example_dict = {}
            else:
                label_r_num = int((line_fields_num - 1) / 2)
                label_r_list, texts_list = line_fields[:label_r_num], line_fields[label_r_num:]
                assert len(texts_list) - len(label_r_list) == 1
                for index, label_r in enumerate(label_r_list):
                    example_dict["text_a"], example_dict["label_ea"] = _decode_sentence(texts_list[index])
                    example_dict["text_b"], example_dict["label_eb"] = _decode_sentence(texts_list[index + 1])
                    example_dict["label_r"] = label_r
                    lines.append(example_dict)
                    example_dict = {}
        # print(lines)
        return lines
    #
    # @classmethod
    # def _read_data(cls, input_file):
    #     """Read data.
    #
    #     Example:
    #         label_r, label_r, seq_1, seq_2, seq_3, timestamp
    #         label_r, seq_1, timestamp
    #     """
    #     df = open(input_file, 'r', encoding='utf8')
    #     lines = []
    #     for line in df:
    #         example_dict = {}
    #         line = json.loads(line)
    #         text, labels = line['text'], line['labels']
    #         endpoint = len(text)
    #         text = text.split(',')[:-1]
    #         labels_e = ['O' for i in range(0, len(','.join(text)))]
    #         example_dict["label_r"] = 'Single'
    #         for label in labels:
    #             if endpoint == label[1]:
    #                 example_dict["label_r"] = label[2]
    #             else:
    #                 start, end, tag = label[0], label[1], label[2]
    #                 labels_e[start:end] = ['I-{}'.format(tag)] * (end - start)
    #                 labels_e[start] = 'B-{}'.format(tag)
    #         if len(text) == 1:
    #             text.append('')
    #         example_dict["text_a"] = ' '.join([c for c in text[0]])
    #         example_dict["text_b"] = ' '.join([c for c in text[1]])
    #         example_dict["label_ea"] = ' '.join(labels_e[:len(text[0])])
    #         if len(text[1]) == 0:
    #             example_dict["label_eb"] = ' '.join([])
    #         else:
    #             example_dict["label_eb"] = ' '.join(labels_e[-len(text[1]):])
    #         lines.append(example_dict)
    #     return lines


class JointProcessor(DataProcessor):
    def _create_example(self, lines, set_type):
        examples = []
        for index, line in enumerate(lines):
            guid = "{}-{}".format(set_type, index)
            text_a = extraction.tokenization.convert_to_unicode(line['text_a'])
            text_b = extraction.tokenization.convert_to_unicode(line['text_b'])
            label_ea = extraction.tokenization.convert_to_unicode(line['label_ea'])
            label_eb = extraction.tokenization.convert_to_unicode(line['label_eb'])
            label_r = extraction.tokenization.convert_to_unicode(line['label_r'])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label_ea=label_ea, label_eb=label_eb,
                                         label_r=label_r))
        return examples

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            # self._read_data(os.path.join(data_dir, "test.txt"), True), "test",
            self._read_data(data_dir, True), "test",
        )

    # def get_test_examples(self, data):
    #     return self._create_example(
    #         data, 'test')

    def get_labeles(self):
        return ['B-actor', 'I-actor', 'B-action', 'I-action', 'B-recipient', 'I-recipient', 'B-object', 'I-object',
                'B-attribute', 'I-attribute', 'O', 'X', '[SEP]', '[CLS]']

    def get_labelrs(self):
        return ['single', 'ori', 'res', 'non', 'con','mto']


    def _create_examples(self, lines, set_type):
        """create examples for the training and val sets"""
        examples = []
        for i, line in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            text_a = extraction.tokenization.convert_to_unicode(line)
            label = extraction.tokenization.convert_to_unicode('non')
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers,
                 initializers, num_labels, seq_length, labels, lengths, dropout_rate=0.9):
        """
        BLSTM-CRF ??????
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM?????????????????????
        :param cell_type: RNN?????????LSTM OR GRU DICNN will be add in feature???
        :param num_layers: RNN?????????
        :param initializers: variable init class
        :param num_labels: ????????????
        :param seq_length: ??????????????????
        :param labels: ????????????
        :param lengths: [batch_size] ??????batch????????????????????????
        """
        self.hidden_unit = hidden_unit
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.dropout_rate = dropout_rate

    def add_blstm_crf_layer(self, crf_only=True):
        """
        blstm-crf??????
        :return:
        """
        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # blstm
            lstm_output = self.blstm_layer(self.embedded_chars)
            # project
            logits = self.project_bilstm_layer(lstm_output)
        # crf
        loss, trans, score = self.crf_layer(logits)

        # trans = trans + rule_matrix
        # CRF decode, pred_ids ????????????????????????????????????
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return (loss, logits, trans, pred_ids, score)

    def _witch_cell(self):
        """
        RNN ??????
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        ??????RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """
        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            rule = [[0.1 for i in range(15)] for j in range(15)]
            I_index_list = [2,4,6,8,10]
            B_index_list = [1,3,5,7,9]
            O_index = 11

            for x in I_index_list:
                rule[O_index][x] = -3
                rule[13][x] = -2
                rule[14][x] = -2
                for y in I_index_list:
                    if x != y:
                        rule[x][y] = -2
                    if x == 6 and y == 8:
                        rule[x][y] == -5
                        rule[y][x] == -5

            for index_b, x in enumerate(B_index_list):
                for index_i, y in enumerate(I_index_list):
                    if index_b != index_i:
                        rule[x][y] = -1
                    if index_b == index_i:
                        rule[x][y] = 0.3

            trans = tf.constant(rule)
            '''
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
            # '''

            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans, log_likelihood


def _truncate_seq_pair(tokens_a, tokens_b, label_ea, label_eb, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        ###TODO: Is there any better ways to truncate
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            label_ea.pop()
        else:
            tokens_b.pop()
            label_eb.pop()


def convert_single_example(example_index, example, labele_list, labelr_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    labele_map, labelr_map = {}, {}
    for index, labele in enumerate(labele_list, 1):
        labele_map[labele] = index
    for index, labelr in enumerate(labelr_list, 1):
        labelr_map[labelr] = index

    text_list_a, labele_list_a = example.text_a.split(' '), example.label_ea.split(' ')
    tokens_a, label_ea = [], []
    # tf.logging.info("text_list_a: %s (lab=%s)", text_list_a, labele_list_a)
    for index, word in enumerate(text_list_a):
        token = tokenizer.tokenize(word)
        if len(token) < 1:
            continue
        tokens_a.extend(token)
        label_ea.extend([labele_list_a[index]] + ['X'] * (len(token) - 1))
    text_list_b, labele_list_b = example.text_b.split(' '), example.label_eb.split(' ')
    tokens_b, label_eb = [], []
    for index, word in enumerate(text_list_b):
        token = tokenizer.tokenize(word)
        if len(token) < 1:
            continue
        tokens_b.extend(token)
        label_eb.extend([labele_list_b[index]] + ['X'] * (len(token) - 1))
    if len(tokens_b) != len(label_eb):
        for index, word in enumerate(text_list_b):
            token = tokenizer.tokenize(word)
            tf.logging.info("token: %s", str(token))
        tf.logging.info("tokens_a: %s label_ea: %s", str(tokens_b), str(label_eb))
    assert len(tokens_a) == len(label_ea)
    assert len(tokens_b) == len(label_eb)
    _truncate_seq_pair(tokens_a=tokens_a, tokens_b=tokens_b, label_ea=label_ea, label_eb=label_eb,
                       max_length=max_seq_length - 3)
    tokens, segment_ids, labele_ids = [], [], []
    tokens.append("[CLS]")
    segment_ids.append(0)
    labele_ids.append(labele_map["[CLS]"])
    # tf.logging.info("tokens_a: %s (lab=%s)", tokens_a, label_ea)
    for token, labele in zip(tokens_a, label_ea):
        tokens.append(token)
        segment_ids.append(0)
        labele_ids.append(labele_map[labele])
    tokens.append("[SEP]")
    segment_ids.append(0)
    labele_ids.append(labele_map["[SEP]"])

    for token, labele in zip(tokens_b, label_eb):
        tokens.append(token)
        segment_ids.append(1)
        labele_ids.append(labele_map[labele])
    tokens.append("[SEP]")
    segment_ids.append(1)
    labele_ids.append(labele_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        labele_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(labele_ids) == max_seq_length

    labelr_id = labelr_map[example.label_r]
    '''
    if example_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s", example.guid)
        tf.logging.info("tokens: %s", " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        tf.logging.info("labele_ids: %s", " ".join([str(x) for x in labele_ids]))
        tf.logging.info("labelr: %s (id=%s)", example.label_r, labelr_id)
    '''
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labele_ids=labele_ids,
        labelr_id=labelr_id)
    return feature


def file_based_covert_examples_to_features(
        examples, labele_list, labelr_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            tf.logging.info("Writing example %d of %d", ex_index, len(examples))

        feature = convert_single_example(ex_index, example, labele_list, labelr_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["labele_ids"] = create_int_feature(feature.labele_ids)
        features["labelr_id"] = create_int_feature([feature.labelr_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to Estimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "labele_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "labelr_id": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            temp = example[name]
            if temp.dtype == tf.int64:
                temp = tf.to_int32(temp)
            example[name] = temp
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        data_set = tf.data.TFRecordDataset(input_file)
        if is_training:
            data_set = data_set.repeat().shuffle(buffer_size=100)

        data_set = data_set.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return data_set
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labele_ids, labelr_id, num_labeles, num_labelrs, use_one_hot_embeddings):
    """Creates a joint learning model."""
    model = extraction.modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.

    ### Task 1
    task1_output_layer = model.get_sequence_output()
    task1_hidden_size = task1_output_layer.shape[-1].value
    ### Task 2
    task2_output_layer = model.get_pooled_output()
    task2_hidden_size = task2_output_layer.shape[-1].value

    ### Weight and bias for task1:
    task1_output_weight = tf.get_variable(
        "task1_output_weights", [num_labeles, task1_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    task1_output_bias = tf.get_variable(
        "task1_output_bias", [num_labeles],
        initializer=tf.zeros_initializer()
    )

    ### Weight and bias for task2:
    task2_output_weight = tf.get_variable(
        "task2_output_weights", [num_labelrs, task2_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    task2_output_bias = tf.get_variable(
        "task2_output_bias", [num_labelrs],
        initializer=tf.zeros_initializer()
    )

    with tf.variable_scope("loss"):
        if is_training:
            task1_output_layer = tf.nn.dropout(task1_output_layer, keep_prob=0.8)
            task2_output_layer = tf.nn.dropout(task2_output_layer, keep_prob=0.9)
        ## task1 calculate
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)  ## [batch_size], current batch
        blstm_crf = BLSTM_CRF(embedded_chars=task1_output_layer, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell,
                              num_layers=FLAGS.num_layers, initializers=tf.truncated_normal_initializer(stddev=0.02),
                              num_labels=num_labeles, seq_length=FLAGS.max_seq_length, labels=labele_ids,
                              lengths=lengths)
        task1_loss, task1_logits, trains, pred_ids, score = blstm_crf.add_blstm_crf_layer(crf_only=False)
        task1_probabilities = tf.nn.softmax(logits=task1_logits, axis=-1)  # return
        task1_predict = tf.argmax(task1_probabilities, axis=-1, output_type=tf.int32)  # return

        ## task2 calculate
        task2_logits = tf.matmul(task2_output_layer, task2_output_weight, transpose_b=True)
        task2_logits = tf.nn.bias_add(task2_logits, task2_output_bias)
        task2_probabilities = tf.nn.softmax(logits=task2_logits, axis=-1)  # return
        task2_log_probs = tf.nn.log_softmax(logits=task2_logits, axis=-1)
        task2_one_hot_labels = tf.one_hot(labelr_id, depth=num_labelrs, dtype=tf.float32)
        ##
        # task2_loss = tf.nn.softmax_cross_entropy_with_logits(labels=task2_one_hot_labels,logits=task2_logits)
        task2_per_example_loss = -tf.reduce_sum(task2_one_hot_labels * task2_log_probs, axis=-1)
        task2_loss = tf.reduce_sum(task2_per_example_loss)  # return
        ###
        task2_predict = tf.argmax(task2_probabilities, axis=-1, output_type=tf.int32)  # return

        ## Joint loss(with weight)
        total_loss = task1_loss + task2_loss  # return

        return total_loss, task1_loss, task2_loss, pred_ids, task2_predict, task1_logits, task2_probabilities, score


def model_fn_builder(bert_config, num_labeles, num_labelrs, init_checkpoint, learning_rate, num_train_steps,
                     num_warmup_steps, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        '''
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("   name = %s, shape = %s", name, features[name].shape)
        '''
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        labele_ids = features["labele_ids"]
        labelr_id = features["labelr_id"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, task1_loss, task2_loss, task1_predict, task2_predict, task1_probabilities, task2_probabilities, score = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, labele_ids, labelr_id, num_labeles,
            num_labelrs, use_one_hot_embeddings)

        ### only task2:
        # total_loss = task1_loss

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            assignment_map, initialized_variable_names = extraction.modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                     init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(labele_ids, task1_predict, num_labeles, labelr_id, task2_predict):
                task1_precision = extraction.tf_metrics.precision(labele_ids, task1_predict, num_labeles,
                                                       [i + 1 for i in range(num_labeles - 4)], average="macro")
                task1_recall = extraction.tf_metrics.recall(labele_ids, task1_predict, num_labeles,
                                                 [i + 1 for i in range(num_labeles - 4)], average="macro")
                task1_f1 = extraction.tf_metrics.f1(labele_ids, task1_predict, num_labeles,
                                         [i + 1 for i in range(num_labeles - 4)], average="macro")
                task1_accuracy = extraction.tf_metrics.sentence_accuracy(labels=labele_ids, predictions=task1_predict)
                # task2_accuracy = extraction.tf.metrics.accuracy(labels=labelr_id, predictions=task2_predict)
                return {
                    "task1_precision": task1_precision,
                    "task1_recall": task1_recall,
                    "task1_f1": task1_f1,
                    "task1_accuracy": task1_accuracy,
                    # "task2_accuracy": task2_accuracy,
                }

            eval_metrics = metric_fn(labele_ids, task1_predict, num_labeles, labelr_id, task2_predict)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)

        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "predict_e": task1_predict,
                    "probilities_e": task1_probabilities,
                    "predict_r": task2_predict,
                    "probilities_r": task2_probabilities,
                    "score": score
                }
            )
        return output_spec

    return model_fn


def main(file):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "joint": JointProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = extraction.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = processors["joint"]()

    labele_list = processor.get_labeles()
    labelr_list = processor.get_labelrs()

    tokenizer = extraction.tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=8,
            per_host_input_for_training=is_per_host))

    # run_config = tf.estimator.RunConfig(
    #    model_dir = FLAGS.output_dir,
    #    save_checkpoints_steps = FLAGS.save_checkpoints_steps)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labeles=len(labele_list) + 1,
        num_labelrs=len(labelr_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    # estimator = tf.estimator.Estimator(
    #    model_fn=model_fn,
    #    config = run_config,
    #    train_batch_size = FLAGS.train_batch_size,
    #    eval_batch_size = FLAGS.eval_batch_size,
    #    predict_batch_size = FLAGS.predict_batch_size)
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_covert_examples_to_features(
            train_examples, labele_list, labelr_list, FLAGS.max_seq_length, tokenizer, train_file)
        '''
        tf.logging.info("***** Running training *****")
        tf.logging.info("   Num examples = %d", len(train_examples))
        tf.logging.info("   Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("   Num steps = %d", num_train_steps)
        '''
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, steps=num_train_steps)
        # full_model_dir = estimator.export_savedmodel(export_dir_base=, serving_input_receiver_fn=serving_input_receiver_fn)

    if FLAGS.do_eval:
        # tf.logging.info(estimator.get_variable_value("loss/crf_loss/transitions:0"))
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_covert_examples_to_features(
            eval_examples, labele_list, labelr_list, FLAGS.max_seq_length, tokenizer, eval_file)
        '''
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("   Num examples = %d", len(eval_examples))
        tf.logging.info("   Batch size = %d", FLAGS.eval_batch_size)
        '''
        eval_steps = None
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        tf.logging.info("***** Eval results *****")
        eval_metric_file = os.path.join(FLAGS.output_dir, "metric.res")
        eval_metric_file = open(eval_metric_file, 'a', encoding='utf8')
        eval_metric_file.write('{}={};'.format('labeled_data', str(len(train_examples))))
        for key in sorted(result.keys()):
            tf.logging.info("   %s = %s", key, str(result[key]))
            eval_metric_file.write('{}={};'.format(key, str(result[key])))
        eval_metric_file.write('\n')

    if FLAGS.do_predict:
        # tf.logging.info(estimator.get_variable_value("loss/crf_loss/transitions:0"))
        #????????????
        predict_examples = processor.get_test_examples(file)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_covert_examples_to_features(
            predict_examples, labele_list, labelr_list, FLAGS.max_seq_length, tokenizer, predict_file)
        tf.logging.info("***** Running prediction *****")
        tf.logging.info("   Num examples = %d", len(predict_examples))
        tf.logging.info("   Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file_lab = os.path.join(FLAGS.data_dir, "label_test.txt")
        output_predict_file_rel = os.path.join(FLAGS.data_dir, "rel_test.txt")
        output_predict_file_crf = os.path.join(FLAGS.data_dir, "crf_test.txt")
        output_predict_file_labeled = os.path.join(FLAGS.data_dir, "labeled_senten_test.txt")
        labeled_writer = tf.gfile.GFile(output_predict_file_labeled, 'w')
        lab_writer = tf.gfile.GFile(output_predict_file_lab, 'w')
        rel_writer = tf.gfile.GFile(output_predict_file_rel, 'w')
        crf_writer = tf.gfile.GFile(output_predict_file_crf, 'w')
        sen_list = []

        for example, precision in zip(predict_examples,result):
            # output_line_lab = '\t'.join(
            #     labele_list[id - 1] + ' ' + str(prob[id]) for id, prob in
            #     zip(precision["predict_e"], precision["probilities_e"]) if id != 0) + '\n'
            # lab_writer.write(output_line_lab)
            #????????????
            label_sen={"text":"",
                       "labels": []}
            if example.text_a == example.text_b:
                label_sen["text"] = example.text_a.split(' ')
            else:
                l = example.text_a.split(' ')
                l.append("???")
                l2 = example.text_b.split(' ')
                l = l+l2
                label_sen["text"] = l
            label_list=[]
            for id, prob in zip(precision["predict_e"], precision["probilities_e"]):
                if id != 0 :
                    label_list.append(labele_list[id - 1])
                    # if labele_list[id - 1] == "[CLS]":
                    #     pass
                    # elif labele_list[id - 1] == "[SEP]":
                    #     break
                    # else:
                    #     label_list.append(labele_list[id - 1])
            label_sen["labels"] = label_list
            ok,encode_line = _encode_one_line(label_sen["text"],label_sen["labels"])
            # labeled_writer.write(json.dumps(encode_line, ensure_ascii=False) + '\n')
            # print(encode_line)
            if ok:
                labeled_writer.write(str(encode_line) + '\n')
                sen_list.append(str(encode_line))

            # output_line_rel = ''.join(
            #     labelr_list[precision["predict_r"] - 1] + " " + str(
            #         precision["probilities_r"][precision["predict_r"]])) + '\n'
            # rel_writer.write(output_line_rel)
            # crf_writer.write(str(precision['score']) + '\n')
        lab_writer.close()
        rel_writer.close()
        crf_writer.close()
        return sen_list



def _encode_one_line(title, labels):
    # ???????????????[CLS]????????????[SEP]
    labels = labels[1:-1]
    # rel = rel.split()[0]
    # if (sentence_pair and rel == 'single') or (not sentence_pair and rel != 'single'):
    #     rel = 'wrong'
    sentence_pair = title.count("???")
    if sentence_pair > 0: # ????????????
        # print(len(labels),len(title))
        if not len(labels) == len(title):
            return False, title
        ok, encode_line = _encode_impl(title, labels)
        if not ok:
            rel = 'wrong'
    else:   # ????????????
        labels = labels[:len(title)]
        ok, encode_line = _encode_impl(title, labels)
        if not ok:
            rel = 'wrong'
    return True, encode_line


def _encode_impl(title, labels):
    res_title, pre_tag = "", ""
    labels = [label.split()[0] for label in labels]
    for c, l in zip(title, labels):
        if l.startswith('B-') and pre_tag == "":
            res_title += "{{" + l[2:] + ":" + c
            pre_tag = l[2:]
        elif l.startswith('B-') and pre_tag != "":
            res_title += "}}{{" + l[2:] + ":" + c
            pre_tag = l[2:]
        elif l == 'O' and pre_tag != "":
            res_title += "}}" + c
            pre_tag = ''
        elif l == 'O' and pre_tag == "":
            res_title +=  c
        elif l.startswith('I-') and pre_tag != l[2:]: # B-X I-Y
            return False, title
        elif l.startswith('I-') and pre_tag == l[2:]:
            res_title +=  c
        elif l == '[SEP]' and c != '???':
            return False, title
        elif l == '[SEP]'and pre_tag != '':
            res_title += "}}" + c
            pre_tag = ''
        elif l == '[SEP]' and pre_tag == '':
            res_title += c
        else:
            res_title += c
    if pre_tag != '':
        res_title += "}}"
    return True, res_title

#??????????????????????????????
def p_run(file):
    global maintext
    maintext = file
    return main(file)
    # tf.app.run()

if __name__ == '__main__':
    # li="text.txt"
    # print(p_run(li))
    sen = main()



# if __name__ == "__main__":
#   flags.mark_flag_as_required("data_dir")
#   flags.mark_flag_as_required("vocab_file")
#   flags.mark_flag_as_required("bert_config_file")
#   flags.mark_flag_as_required("output_dir")
#   tf.app.run()
