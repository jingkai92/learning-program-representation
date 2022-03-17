import re
import sys
from tqdm import tqdm
from nltk.corpus import stopwords


class WordTokenizer:
    def __init__(self, data, remove_sw=False):
        self.data = data  # Try to have a list of sentence
        self.hash_pattern = re.compile("^[A-Fa-f0-9]{40}$")
        self.alphanum_pattern = re.compile("^\d+\w")
        self.version_num_pattern = re.compile("^(\d+\.)?(\d+\.)?(\*|\d+)$")
        self.url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[#$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.special_tokens = self.set_special_tokens()
        self.remove_stop_word = remove_sw

    @staticmethod
    def set_special_tokens():
        """
        Set up a dictionary of special tokens. Word that are one time usage are commonly dropped.
        If we did not preserve these one time tokens, the meaning of the sentence might not be capture
        :return:
        """
        tmp = dict()
        tmp['HASHID'] = "HASHID"
        tmp['NUM'] = "NUM"
        tmp['ALPHANUM'] = "ALPHANUM"
        tmp['VERSIONNUM'] = "VERSIONNUM"
        tmp['URL'] = "URL"
        return tmp

    def tokenize_data(self):
        """
        More sophicated tokenizing on this
        :return:
        """
        tokenized_msgs = []
        for msg in tqdm(self.data, file=sys.stdout):
            tmsg = self._tokenize(msg)
            if tmsg:
                tokenized_msgs.append(tmsg)
        return tokenized_msgs

    def tokenize_commits(self, commit_list, disable=False):
        """
        Tokenise Commit, same as tokenize_data
        :param commit_list:
        :param disable:
        :return:
        """
        tokenized_msgs = []
        for msg in tqdm(commit_list, disable=disable):
            tmsg = self._tokenize(msg)
            if tmsg:
                if self.remove_stop_word:
                    stops = set(stopwords.words("english"))
                    tmsg = [w for w in tmsg if not w in stops]
                tokenized_msgs.append(tmsg)
        return tokenized_msgs

    def _tokenize(self, msg):
        """
        Transform a message into a list of token
        E.g lib: fix a SEGV in list_possible_events() The bug has been introduced in commit -->
        [lib,:, fix, a, SEGV, in, list_possible_events(), The, bug, has, been, introduced, in, commit]
        :return:
        """
        clean_msg_list = self._clean_tokens(msg)
        token_list = []
        for token in clean_msg_list:
            new_token = self._handle_token(token)
            if new_token not in self.special_tokens:
                new_token = new_token.lower()
            token_list.append(new_token)

        clean_token_list = " ".join(token_list).split()
        return clean_token_list

    def _clean_tokens(self, msg):
        """
        Clean the tokens of punctuanction and special symbol.
        We also detect the version number and some alphanumber symbol and exchange them with symbol
        Reason is that after you remove punc, you cannot detect them anymore
        :param msg:
        :return:
        """
        clean_token_list = []
        msg_list = msg.split()
        for token in msg_list:
            new_token = self._clean_token(token)
            if new_token not in self.special_tokens:
                new_token = new_token.lower()
            clean_token_list.append(new_token)

        clean_token_list = " ".join(clean_token_list).split()
        return clean_token_list

    def _handle_token(self, token):
        """
        Some rules for handling token.
        :param token:
        :return:
        """
        token = self._is_hash(token)
        token = self._is_number(token)
        token = self._is_alphanumeric(token)  # This should be a wild card, last. If it is not the rest, then you check
        return token

    def _clean_token(self, token):
        """
        Clean the tokens. You should remove all other unnecessary words. But I wish to preserve some version number and
        maybe some other stuff
        :param token:
        :return:
        """
        # Reason being that after removing punctuation, you cannot detect version number and url
        if self.version_num_pattern.match(token):
            return self.special_tokens["VERSIONNUM"]
        elif self.url_pattern.match(token) or self.url_pattern.search(token):
            return self.special_tokens["URL"]
        else:
            token = re.sub(r'[^\w\s]', ' ', token)
            token = self._has_underscore(token)
            return token.strip()

    def _is_hash(self, token):
        """
        Check if token is hash, if it is hash, replace hash with HASH_ID
        :param token:
        :return:
        """
        res = self.hash_pattern.match(token)
        if res:
            return self.special_tokens["HASHID"]
        else:
            return token

    def _is_number(self, token):
        """
        Check if token is number, if it is number, replace hash with NUM
        :param token:
        :return:
        """
        try:
            tint = int(token)
            return self.special_tokens["NUM"]
        except ValueError as ve:
            return token

    @staticmethod
    def _has_underscore(token):
        """
        Check if the token has underscore, if it has underscore, split it into multiple words
        :param token:
        :return:
        """
        if "_" in token:
            token = token.replace("_", " ")
            return token
        else:
            return token

    def _is_alphanumeric(self, token):
        """

        :param token:
        :return:
        """
        res = self.alphanum_pattern.match(token)
        if res:
            return self.special_tokens["ALPHANUM"]
        else:
            return token
