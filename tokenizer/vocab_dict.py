import operator
import pickle
from enum import Enum
import youtokentome as yttm


class TokenizationConstant(Enum):
    SOS = "0"
    EOS = "1"
    OOV = "2"
    PAD = "3"


class VocabDictBase:
    def __init__(self, name, file_name):
        self.name = name
        self.file_name = file_name

    def load(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def is_bpe(self):
        return False


class VocabDict(VocabDictBase):
    def __init__(self, name, file_name):
        VocabDictBase.__init__(self, name, file_name)
        self.word2index = {"SOS": 0, "EOS": 1, "OOV": 2, "PAD": 3}
        self.word2count = {"SOS": 1, "EOS": 1, "OOV": 1, "PAD": 1}
        self.index2word = {0: "SOS", 1: "EOS", 2: "OOV", 3: "PAD"}
        self.n_words = 4  # Count SOS and EOS

    def add_sentence(self, sentence):
        """
        Add the word in the sentence into the dictionary
        :param sentence: Sentence
        :return: NIL
        """
        assert type(sentence) == str
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        Add a single word into the dictionary
        :param word: Word
        :return:
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def get_w2i(self, word):
        """
        Get the index of the word. If it does not exists, return OOV number
        :param word: Targetting Word
        :return: return index or OOV
        """
        try:
            return self.word2index[word]
        except KeyError as ke:
            return self.word2index["OOV"]

    def get_i2w(self, index):
        """
        Get the word by using the index
        :param index: Target index
        :return: return word
        """
        return self.index2word[index]

    def load(self):
        """
        Load self from self.file_name
        :return:
        """
        f = open(self.file_name, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self):
        """
        Save self to self.file_name using pickle
        :return:
        """
        f = open(self.file_name, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def save2(self, path):
        """
        Save self to self.file_name using pickle
        :return:
        """
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def limit_vocab(self, limiting_size):
        """
        Limit the vocabulary to a certain size
        self.word2index = {"SOS": 0, "EOS": 1, "OOV": 2, "PAD": 3, "<EMPTY_CODE>": ??}
        self.word2count = {"SOS": 1, "EOS": 1, "OOV": 1, "PAD": 1}
        :param limiting_size: Limiting Size
        :return: NIL, inplace method
        """
        sorted_word2count = sorted(self.word2count.items(), key=operator.itemgetter(1), reverse=True)
        limited_w2c = dict(sorted_word2count[:limiting_size])
        limited_w2c['SOS'] = 1
        limited_w2c['EOS'] = 1
        limited_w2c['OOV'] = 1
        limited_w2c['PAD'] = 1
        limited_w2c['<EMPTY_CODE>'] = 1
        assert 'SOS' in limited_w2c and 'EOS' in limited_w2c and 'OOV' in limited_w2c and \
               'PAD' in limited_w2c and "<EMPTY_CODE>" in limited_w2c
        self.word2count = limited_w2c
        self._sync_w2i_and_w2c()

    def _sync_w2i_and_w2c(self):
        """
        Sync self.word2count and self.word2index
        :return:
        """
        new_word2index = dict()
        for key, value in self.word2index.items():
            if key in self.word2count:
                new_word2index[key] = len(new_word2index)
        self.word2index = new_word2index
        assert len(self.word2index) == len(self.word2count)

    def convert_ids_to_sentence(self, ids):
        """
        Convert a list of IDs into a sentence
        :param ids: List of IDs
        :return: Return a sentence in String
        """
        word_list = []
        for id in ids:
            word = self.get_i2w(id)
            word_list.append(word)
        return " ".join(word_list)

    def __len__(self):
        """
        Length of the Tokenizer = Length of W2i
        :return:
        """
        return len(self.word2index)

    def __str__(self):
        return str(self.word2index)

    def vocab_size(self):
        """
        Return the vocab size
        :return:
        """
        return len(self.word2index)


class BPEVocabDict(VocabDictBase):
    def __init__(self, name, file_name):
        VocabDictBase.__init__(self, name, file_name)
        self.bpe_model = None

    def load(self):
        """
        Load the BPE Model
        :return:
        """
        self.bpe_model = yttm.BPE(model=self.file_name)

    def convert_sent_to_ids(self, sent, eos=False):
        enc_seqs = self.bpe_model.encode(sent, output_type=yttm.OutputType.ID,
                                         bos=eos, eos=eos)
        return enc_seqs

    def __len__(self):
        """
        Size of the BPE Model
        :return: Return Size of BPE Model
        """
        return self.bpe_model.vocab_size()

    def vocab_size(self):
        """
        Return the vocab size
        :return:
        """
        return self.bpe_model.vocab_size()

    def is_bpe(self):
        return True

    def convert_ids_to_sentence(self, seqs):
        return self.bpe_model.decode(seqs)
