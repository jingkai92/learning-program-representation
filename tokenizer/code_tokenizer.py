import re
import sys
from tqdm import tqdm
from pygments import lex
from pygments.token import Token
from pygments.lexers import get_lexer_by_name


class CodeTokenizer:
    def __init__(self, data, lang, tlevel):
        """
        CodeTokenizer - Tokenize source code by their tlevel
        :param data: A list of data, right now we dont really use this
        :param lang: Language, Support only C right now
        :param tlevel: Tokenizer level - See README.md for more information
        """
        self.data = data  # Try to have a list of sentence
        self.tlevel = tlevel
        self.hash_pattern = re.compile("^[A-Fa-f0-9]{40}$")
        self.alphanum_pattern = re.compile("^\d+\w")
        self.version_num_pattern = re.compile("^(\d+\.)?(\d+\.)?(\*|\d+)$")
        self.url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[#$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.special_tokens = self.set_special_tokens()
        self.decimal_pattern = re.compile(r'^\d+(?:[,.]\d*)?$')
        self.lexer = get_lexer_by_name(lang, stripall=True)
        self.variable_dict = dict()
        self.method_dict = dict()
        self.excluded_list = ["__asm__", "__volatile__", "noinline", "__kprobes", "PHPAPI",
                              "asmlinkage", "True", "False", "KERN_INFO", "KERN_WARNING", "KERN_ERR"]

    @staticmethod
    def set_special_tokens():
        """
        Set up a dictionary of special tokens. Word that are one time usage are commonly dropped.
        If we did not preserve these one time tokens, the meaning of the sentence might not be capture
        :return:
        """
        tmp = dict()
        tmp['HASHID'] = "HASHID"
        tmp['NUMLITERAL'] = "NUMLITERAL"
        tmp['ALPHANUM'] = "ALPHANUM"
        tmp['VERSIONNUM'] = "VERSIONNUM"
        tmp['URL'] = "URL"
        tmp['HEXLITERAL'] = "HEXLITERAL"
        tmp['STRINGLITERAL'] = "STRINGLITERAL"
        return tmp

    def tokenize_data(self):
        """
        This is the tokenizing function for a whole list of data. Actually, I
        did not use this much. Only use the tokenize(code) more often as it
        is more useful
        :return: Return a whole list of list of tokenized source code
        """
        tokenized_msgs = []
        for msg in tqdm(self.data, file=sys.stdout):
            tmsg = self._tokenize(msg)
            if tmsg:
                tokenized_msgs.append(tmsg)
        return tokenized_msgs

    def tokenize(self, code):
        """
        Tokenize a single string of code
        :param code: A source code string
        :return: Return a string
        """
        # Clean the variable/method dictionary so that it is fresh for one function
        self.variable_dict = dict()
        self.method_dict = dict()
        # Parsing, Cleaning, and removing comments are commonly among all tlevel
        token_list = list(lex(code, self.lexer))
        token_list = self._clean_empty(token_list)  # Strip Whitespaces
        tokens = self._remove_comments(token_list)  # Remove Comments from the token lists
        if self.tlevel == 't1':  # Basically, nothing else
            pass
        if self.tlevel == 't2':
            tokens = self._tokenize_t2(tokens)
        if self.tlevel == 't3':
            tokens = self._tokenize_t3(tokens)
        if self.tlevel == 't4':
            tokens = self._tokenize_t4(tokens)

        token_strings = [x[1].lower().replace("\n", "<slash_n>")
                         if x[1] not in self.special_tokens else x[1] for x in tokens]
        return " ".join(token_strings)

    def _tokenize_t2(self, token_list):
        """
        T2 Tokenizing Level - See README for more information
        :param token_list: A list of token produced by the pygments
        :return: Return T1 Level of Tokenized list of tokens
        """
        codes = self._split_var(token_list)
        return codes

    def _tokenize_t3(self, token_list):
        """
        T3 Tokenizing Level - See README for more information
        :param token_list: A list of token produced by the pygments
        :return: Return T1 Level of Tokenized list of tokens
        """
        codes = self._replace_tokens(token_list)  # Remove Literal Tokens with Placeholders
        codes = self._split_var(codes)
        return codes

    def _tokenize_t4(self, token_list):
        """
        T4 Tokenizing Level - See README for more information
        :param token_list: A list of token produced by the pygments
        :return: Return T1 Level of Tokenized list of tokens
        """
        codes = self._replace_tokens(token_list)  # Remove Literal Tokens with Placeholders
        codes = self._replace_identifier(codes)
        return codes

    def _tokenize(self, code):
        """
        Transform a message into a list of token
        E.g static char	elsieid[] = "@(#)emkdir.c	8.21"; -->
        [static, char, elsieid, [, ], =, ", @, (, #, ), emkdir, ., c, 8, ., 2, 1]
        :return:
        """
        token_list = list(lex(code, self.lexer))
        token_list = self._clean_empty(token_list)  # Remove Empty (" ", "\t", "\n")
        nc_code = self._remove_comments(token_list)  # Remove Comments from the token lists
        codes = self._replace_tokens(nc_code)  # Remove Literal Tokens with Placeholders
        codes = self._split_var(codes)
        codes = self._clean_string_literal(codes)
        lower_cases = [x[1].lower() for x in codes if x not in self.special_tokens]
        string_only = [x.replace("\n", "") for x in lower_cases]
        return string_only

    def get_names(self, code):
        """
        Get only the name from the code
        :param code: Code String
        :return: Return a list of token that are only names
        """
        token_list = list(lex(code, self.lexer))
        tmp = []
        for token in token_list:
            if token[0] == Token.Name or token[0] == Token.Name.Function or token[0] == Token.Name.Class:
                tmp.append(token)
        token_strings = [x[1].lower().replace("\n", "<slash_n>")
                         if x[1] not in self.special_tokens else x[1] for x in tmp]
        return " ".join(token_strings)

    def remove_keywords(self, code):
        """
        Remove the keywords from a code string
        :param code: The code string
        :return: Return a string without any keywords
        """
        token_list = list(lex(code, self.lexer))
        tmp = []
        for token in token_list:
            if token[0] == Token.Keywords:
                continue
            tmp.append(token)
        token_strings = [x[1].lower().replace("\n", "<slash_n>")
                         if x[1] not in self.special_tokens else x[1] for x in tmp]
        return " ".join(token_strings)

    @staticmethod
    def _clean_empty(token_list):
        """
        Remove empty tokens, Put first to reduce complexity
        :param token_list:
        :return:
        """
        tmp = []
        for token in token_list:
            token_stripped = token[1].strip()
            if not token_stripped:
                continue
            tmp.append(token)
        return tmp

    @staticmethod
    def _remove_comments(token_list):
        """
        Remove comments from the code
        :return:
        """
        no_comments = []
        for token in token_list:
            if token[0] == Token.Comment.Single or token[0] == Token.Comment.Multiline:
                continue
            if token[0] == Token.Comment.PreprocFile or token[0] == Token.Comment.Preproc:
                continue
            no_comments.append(token)
        return no_comments

    def _replace_identifier(self, token_list):
        """
        The input is a list of token and our job here is to transform all the identifier to either
        mx for method/function name or vx for variable name.
        We will standardize all the method/function to mx for convienience
        :param token_list: A list of tokens
        :return: Return
        """
        cleaned_tokens = []
        for x, token in enumerate(token_list):
            # print(token)
            # Excluded List for special compiler modifier - Mostly Rare
            if token[1] in self.excluded_list:
                cleaned_tokens.append(token)
                continue
            if token[0] == Token.Name.Function:
                abs_name = self._get_abstracted_method_name(token[1])
                cleaned_tokens.append((token[0], abs_name))
            elif token[0] == Token.Name.Class:
                abs_name = self._get_abstracted_variable_name(token[1])
                cleaned_tokens.append((token[0], abs_name))
            elif token[0] == Token.Name:
                if x == len(token_list) - 1:
                    abs_name = self._get_abstracted_variable_name(token[1])
                    cleaned_tokens.append((token[0], abs_name))
                    continue
                next_token = token_list[x+1]
                if next_token[1] in ["=", "+", "-", ";", "/", ")", "!", "|", '}',
                                     ',', "<", ">", "*", '[', ']', ".", "?", "&",
                                     "{", '"', "%", "^", ":"]:  # Variable
                    abs_name = self._get_abstracted_variable_name(token[1])
                    cleaned_tokens.append((token[0], abs_name))
                    continue
                elif next_token[1] in ["("]:  # Function/Method
                    abs_name = self._get_abstracted_method_name(token[1])
                    cleaned_tokens.append((token[0], abs_name))
                    continue
                # this is probably a return type or custom data-type, such as gboolean
                elif next_token[0] in [Token.Name.Function, Token.Name, Token.Keyword,
                                        # This is a bug, document_length-5 should be parsed as name, operator and integer
                                        # see readme
                                       Token.Literal.Number.Integer, Token.Literal.Number.Float,
                                       Token.Literal.Number.Hex,
                                        # Probably some keyword infront of return type, such as av_cold, asmlinkage
                                       Token.Keyword.Type, Token.Keyword.Reserved,
                                       # Struct thingy, ## possible \\\n due to next line and \ for continue on next line
                                       Token.Name.Label, Token.Text, Token.Name.Builtin, Token.Comment]:  # Variable
                    abs_name = self._get_abstracted_variable_name(token[1])
                    cleaned_tokens.append((token[0], abs_name))
                    continue
                elif next_token[0] == Token.Error:
                    cleaned_tokens.append(token)
                    continue
                else:
                    print("Next Token:", next_token)
                    raise Exception("Unknown Next Token")
            else:
                cleaned_tokens.append(token)
        return cleaned_tokens

    def _get_abstracted_method_name(self, name):
        """
        Retrieve the respective function/method name for the input name
        :param name: Name of the function
        :return: Return the abstracted name
        """
        if name in self.method_dict:
            return self.method_dict[name]
        self.method_dict[name] = 'm' + str(len(self.method_dict))
        return self.method_dict[name]

    def _get_abstracted_variable_name(self, name):
        """
        Retrieve the respective variable name for the input name
        :param name: Name of the function
        :return: Return the abstracted name
        """
        if name in self.variable_dict:
            return self.variable_dict[name]
        self.variable_dict[name] = 'v' + str(len(self.variable_dict))
        return self.variable_dict[name]

    def _replace_tokens(self, token_list):
        """
        Replace Literal like 1, 2 ,3 to NUM, etc
        :param token_list: A list of processed tokens
        :return: Return the tokens that has been replaced by placeholder
        """
        cleaned_tokens = []
        for token in token_list:
            if token[0] == Token.Literal.Number.Integer:
                cleaned_tokens.append((Token.Literal.Number.Integer, self.special_tokens["NUMLITERAL"]))
                continue
            elif token[0] == Token.Literal.Number.Float:
                cleaned_tokens.append((Token.Literal.Number.Float, self.special_tokens["NUMLITERAL"]))
                continue
            elif token[0] == Token.Literal.Number.Hex:
                cleaned_tokens.append((Token.Literal.Number.Hex, self.special_tokens["HEXLITERAL"]))
                continue
            elif token[0] == Token.Literal.String and not token[1] == '"':
                cleaned_tokens.append((Token.Literal.String, self.special_tokens["STRINGLITERAL"]))
                continue
            else:
                cleaned_tokens.append(token)
        return cleaned_tokens

    def _split_var(self, token_list):
        """
        Split Variable name and Function Names
        :param token_list:
        :return:
        """
        cleaned_tokens = []
        for token in token_list:
            if token[0] == Token.Name or token[0] == Token.Name.Function or token[0] == Token.Name.Class:
                if "_" in token[1]:
                    var_splitted = self._split_by_underscore(token[1], Token.Name)
                    cleaned_tokens.extend(var_splitted)
                    continue
                else:  # check for camel case
                    camel_words = self.camel_case_split(token[1], Token.Name)
                    cleaned_tokens.extend(camel_words)
            elif token[0] == Token.Name.Namespace:
                dots_splitted = self._split_by_dots(token[1], Token.Name)
                cleaned_tokens.extend(dots_splitted)
                continue
            else:
                cleaned_tokens.append(token)
        return cleaned_tokens

    @staticmethod
    def _split_by_dots(token, token_type):
        """
        Split the namespace by dots
        :param token:
        :param token_type:
        :return:
        """
        var_splitted = token.split(".")
        var_splitted = [(token_type, x) for x in var_splitted if x]
        return var_splitted

    def _clean_string_literal(self, token_list):
        """
        Split String Literal into smaller pieces, by either space, underscore or camelcase
        Remove Num String Literal
        :param token_list:
        :return:
        """
        space_splitted = self._split_string_literal(token_list)
        remove_num_string = self._remove_num_string(space_splitted)
        underscore_splitted = self._underscore_split(remove_num_string)
        cc_splitted = self._camelcase_split(underscore_splitted)
        return cc_splitted

    def _split_string_literal(self, token_list):
        """

        :param token_list:
        :return:
        """
        cleaned_tokens = []
        for token in token_list:
            if token[0] == Token.Literal.String:
                var_split_by_space = self._split_by_space(token[1], Token.Literal.String)
                cleaned_tokens.extend(var_split_by_space)
            else:
                cleaned_tokens.append(token)
        return cleaned_tokens

    @staticmethod
    def _split_by_underscore(token, token_type):
        """
        Split the given token by _
        :param token: Token string
        :return:
        """
        var_splitted = token.split("_")
        var_splitted = [(token_type, x) for x in var_splitted if x]
        return var_splitted

    @staticmethod
    def _split_by_space(token, token_type):
        """
        Split the given token by _
        :param token: Token string
        :return:
        """
        var_splitted = token.split()
        var_splitted = [(token_type, x) for x in var_splitted if x]
        return var_splitted

    @staticmethod
    def camel_case_split(identifier, token_type):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        mg = [(token_type, m.group(0)) for m in matches]
        return mg

    def _underscore_split(self, token_list):
        """

        :param token_type:
        :return:
        """
        cleaned_tokens = []
        for token in token_list:
            if token[0] == Token.Literal.String:
                if "_" in token[1]:
                    var_splitted = self._split_by_underscore(token[1], Token.Literal.String)
                    cleaned_tokens.extend(var_splitted)
                    continue
                else:
                    cleaned_tokens.append(token)
            else:
                cleaned_tokens.append(token)
        return cleaned_tokens

    def _camelcase_split(self, token_list):
        """
        Split the Variable name by camelcase
        :param token_list:
        :return:
        """
        cleaned_tokens = []
        for token in token_list:
            if token[0] == Token.Literal.String:
                camel_words = self.camel_case_split(token[1], Token.Name)
                cleaned_tokens.extend(camel_words)
            else:
                cleaned_tokens.append(token)
        return cleaned_tokens

    def _remove_num_string(self, token_list):
        """
        Remove the string that contain integer/number LITERAL
        :param token_list: A List of Tokens
        :return:
        """
        cleaned_tokens = []
        for token in token_list:
            if token[0] == Token.Literal.String:
                if token[1].isdigit():
                    cleaned_tokens.append((token[0], self.special_tokens["NUM"]))
                    continue
                elif self.decimal_pattern.match(token[1]):
                    cleaned_tokens.append((token[0], self.special_tokens["NUM"]))
                    continue
                else:
                    cleaned_tokens.append(token)
            else:
                cleaned_tokens.append(token)
        return cleaned_tokens
