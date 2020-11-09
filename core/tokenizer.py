# Токенизатор текста для Discord
# by Sergree
# https://github.com/sergree

import numpy as np
import re
import pickle
import cyrtranslit


class Tokenizer:

    entity_to_word = {
        0: "_URL_",
        1: "_MY_MENTION_",
        2: "_MEMBER_MENTION_",
        3: "_CHANNEL_MENTION_",
        4: "_ROLE_MENTION_",
        5: "_CUSTOM_EMOJI_",
        6: "_ANIMATED_CUSTOM_EMOJI_",
    }

    def __init__(self):
        self.index_to_word = {}
        self.word_to_index = {}

    def fill_index_to_word(self):
        self.index_to_word = {value: key for key, value in self.word_to_index.items()}

    def load_vocab_from_file(self, fname):
        with open(fname, "rb") as file:
            self.word_to_index = pickle.load(file)
            self.fill_index_to_word()

    @staticmethod
    def trigramize(word):
        trigrams = []
        for idx, char in enumerate(word):
            if idx == 0:
                first = "*"
            else:
                first = word[idx - 1]
            second = char
            if idx == len(word) - 1:
                third = "*"
            else:
                third = word[idx + 1]
            trigrams.append(first + second + third)
        return trigrams

    def tokenize(self, content, author_id, my_id=0):
        tuples = re.findall(
            r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|"
            r"(<@!?" + str(my_id) + r">)|"
            r"(<@!?\d{16,20}>)|"
            r"(<#\d{16,20}>)|"
            r"(<@&\d{16,20}>)|"
            r"(<:\w{1,32}:\d{16,20}>)|"
            r"(<[a]:\w{1,32}:\d{16,20}>)|"
            r"(@everyone|@here)|"
            r"([^\d\W]+)|"
            r"(.)",
            content,
            re.UNICODE,
        )
        result = []
        if author_id == my_id:
            result.append("_MY_MESSAGE_BEGIN_")
        else:
            result.append("_NOT_MY_MESSAGE_BEGIN_")
        for tup in tuples:
            for idx, item in enumerate(tup):
                if item:
                    if idx <= 6:
                        result.append(self.entity_to_word[idx])
                    elif idx == 7:
                        result.append(item)
                    elif idx == 8:
                        if item.isupper():
                            result.append("_CAPS_")
                        elif item[0].isupper():
                            result.append("_SHIFT_")
                        trigrams = self.trigramize(
                            cyrtranslit.to_cyrillic(item.lower(), "ru")
                        )
                        result.extend(trigrams)
                    else:
                        result.append(item)
        result.append("_MESSAGE_END_")
        return result

    def get_index_by_word(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index["_UNK_"]

    def encode_input(self, messages, me, max_len=64):
        encoder_input_data = np.zeros((1, max_len), dtype="uint16")
        my_id = me.id
        tokenized_input = []
        for message in messages:
            tokenized_input.extend(
                self.tokenize(message.content, message.author.id, my_id=my_id)
            )
        if len(tokenized_input) > max_len:
            tokenized_input = tokenized_input[-max_len:]
        for idx, token in enumerate(tokenized_input):
            encoder_input_data[0, idx] = self.get_index_by_word(token)
        return encoder_input_data

    def decode_output(self, discord_client, input_messages, tensor):
        tokens = []
        for idx in tensor:
            tokens.append(self.index_to_word[idx])
        message = ""
        caps_active = False
        shift_active = False
        last_token = None
        for token in tokens:
            reset_shift_and_caps = True
            if token == " ":
                if last_token != " ":
                    message += token
            elif len(token) < 3:
                message += token
            elif len(token) == 3:
                if shift_active or caps_active:
                    message += token[1].upper()
                    shift_active = False
                    reset_shift_and_caps = False
                else:
                    message += token[1]
                    reset_shift_and_caps = False
            elif token == "_SHIFT_":
                shift_active = True
                reset_shift_and_caps = False
            elif token == "_CAPS_":
                caps_active = True
                reset_shift_and_caps = False
            elif token in ["_CUSTOM_EMOJI_", "_ANIMATED_CUSTOM_EMOJI_"]:
                message += discord_client.random_emoji()
            elif token == "_MY_MENTION_":
                message += discord_client.user.mention
            elif token == "_MEMBER_MENTION_":
                other_members = []
                for input_message in input_messages:
                    if input_message.author != discord_client.user:
                        other_members.append(input_message.author.mention)
                if len(other_members) > 0:
                    message += other_members[-1]
            last_token = token
            if reset_shift_and_caps:
                caps_active = False
                shift_active = False
        return message, len(tokens)
