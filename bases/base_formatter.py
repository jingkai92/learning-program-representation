from tasks.common.datapoint import Datapoint
from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.vocab_dict import TokenizationConstant
from factory.collate_factory import CollateFactory


class BaseFormatter:
    def __init__(self, config, name):
        """
        Base Formatter - For standard formatting
        :param config: Configuration Object
        :param name: Name of the formatter
        """
        self.name = name
        self.config = config
        self.datapoint_class = Datapoint
        self.disable_tqdm = config.disable_tqdm
        self.collate_fn = CollateFactory().get_collate_fn(self.config)
        collate_name = self.collate_fn.__name__ if self.collate_fn is not None else "NIL"
        self.config.logger.info("Collate: %s" % collate_name)
        self.t3_parser = CodeTokenizer(data=[], lang="C", tlevel='t3')
        self.missing_word_count = 0

    def tokenize_sentence(self, sentence, tokenizer, eos=True):
        """
        Tokenize a sentence without any padding since we will pad it during collates
        :param eos: Specify if you want EOS and SOS on the start of the sentence
        :param sentence: Sentence to be tokenized. Single Sentence
        :param tokenizer: Tokenizer object that contains the word2index dictionary
        :return: Return the tokenizer words
        """
        if eos:
            enc_seqs = self._tokenize_sentence_with_eos(sentence, tokenizer)
        else:
            enc_seqs = self._tokenize_sentence_without_eos(sentence, tokenizer)
        enc_seqs_len = len(enc_seqs)
        return enc_seqs, enc_seqs_len

    def _tokenize_sentence_with_eos(self, sent, tokenizer):
        """
        Tokenizer a single sentence
        :param sent: A Single Sentence in String Format
        :return: Return a list of tokens
        """
        if tokenizer.is_bpe():
            return tokenizer.convert_sent_to_ids(sent, eos=True)
        tokens = [tokenizer.get_w2i("SOS")]
        item_split = sent.split()
        for word in item_split:
            idx = tokenizer.get_w2i(word)
            tokens.append(idx)
            if idx == int(TokenizationConstant.OOV.value):
                self.missing_word_count += 1
        tokens.append(tokenizer.get_w2i("EOS"))
        return tokens

    def _tokenize_sentence_without_eos(self, sent, tokenizer):
        """
        Tokenizer a single sentence
        :param sent: A Single Sentence in String Format
        :return: Return a list of tokens
        """
        if tokenizer.is_bpe():
            return tokenizer.convert_sent_to_ids(sent, eos=False)
        tokens = []
        item_split = sent.split()
        for word in item_split:
            idx = tokenizer.get_w2i(word)
            tokens.append(idx)
            if idx == int(TokenizationConstant.OOV.value):
                self.missing_word_count += 1
        return tokens

    @staticmethod
    def pad_or_trim(tokenized_sequences, padding_num=0, max_len=0):
        """
        Either you pad the sequences or you trim them so that they will be at the same length of
        config.max_sequence_length
        :param tokenized_sequences: Sequence to be pad or trim. It is a list
        :param padding_num: Number for padding
        :param max_len: Max Sequence Length
        :return: Return the padded or trimmed single_vocab_sequence
        """
        assert max_len > 0
        trim_seq = tokenized_sequences
        if len(tokenized_sequences) > max_len:
            trim_seq = tokenized_sequences[:max_len]
        elif len(tokenized_sequences) < max_len:
            diff = max_len - len(tokenized_sequences)
            for i in range(diff):
                tokenized_sequences.append(padding_num)
        return trim_seq

    def _pad_seqlist(self, seqlist, max_len):
        """
        Pad the seqlist into max Sequence length
        :param seqlist:
                :param max_len: Maximum Length for Sequences
        :return:
        """
        padded_enc_seqs = []
        lens = []
        for item in seqlist:
            lens.append(len(item))
            item = self.pad_or_trim(item, padding_num=int(TokenizationConstant.PAD.value),
                                    max_len=max_len)
            if not (item[-1] == int(TokenizationConstant.EOS.value) or item[-1] == int(TokenizationConstant.PAD.value)):
                item[-1] = int(TokenizationConstant.EOS.value)
            padded_enc_seqs.append(item)
        return padded_enc_seqs, lens
