from .trie import TokenTrieNode, FunctionTrie
from .validation import Functions
from enum import Enum, auto
from .tokeniser import CustomQwenTokeniser
import numpy as np


class JsonState(Enum):
    START = auto()
    EXPECTED_NAME_KEY = auto()
    EXPECTED_FUNC_KEY = auto()
    EXPECTED_PARAM_KEY = auto()
    EXPECTED_VAL_KEY = auto()
    EXPECTED_VAL_VALUE = auto()
    EXPECTED_CLOSE = auto()


class FSM():
    def __init__(self, trie: FunctionTrie, tokeniser: CustomQwenTokeniser) -> None:
        self.cur_state = JsonState.START
        self.trie_root: TokenTrieNode = trie.root
        self.cur_trie_node: TokenTrieNode = self.trie_root
        self.tokeniser: CustomQwenTokeniser = tokeniser

        self.cur_func: Functions
        self.value_counter: int = 0
        self.func_list = trie.function_list

    def transition(self, last_token: int) -> None:
        if self.cur_state == JsonState.START:
            self.cur_state = JsonState.EXPECTED_NAME_KEY
        elif self.cur_state == JsonState.EXPECTED_NAME_KEY:
            self.cur_state = JsonState.EXPECTED_FUNC_KEY
        elif self.cur_state == JsonState.EXPECTED_FUNC_KEY:
            if self.cur_trie_node.func_name is not None:
                self.cur_func = self.get_function_link(
                    self.cur_trie_node.func_name)
                self.cur_state = JsonState.EXPECTED_PARAM_KEY
                self.cur_trie_node = self.trie_root
            else:
                self.cur_trie_node = self.cur_trie_node.children[last_token]
        elif self.cur_state == JsonState.EXPECTED_PARAM_KEY:
            self.cur_state = JsonState.EXPECTED_VAL_KEY
        elif self.cur_state == JsonState.EXPECTED_VAL_KEY:
            self.value_counter += 1
            if self.value_counter != len(self.cur_func.parameters) - 1:
                self.cur_state = JsonState.EXPECTED_VAL_KEY
            else:
                self.cur_state = JsonState.EXPECTED_CLOSE
                self.value_counter = 0
        elif self.cur_state == JsonState.EXPECTED_CLOSE:
            self.cur_state = JsonState.START

    def get_allowed_tokens(self) -> list[int]:
        allowed_tokens: list[int] = []
        if self.cur_state == JsonState.START:
            return self.tokeniser.encode("{")
        elif self.cur_state == JsonState.EXPECTED_NAME_KEY:
            return self.tokeniser.encode('"name":"')
        elif self.cur_state == JsonState.EXPECTED_FUNC_KEY:
            for child in self.cur_trie_node.children:
                allowed_tokens.append(child)
        elif self.cur_state == JsonState.EXPECTED_PARAM_KEY:
            return self.tokeniser.encode('""parameters":{')
        elif self.cur_state == JsonState.EXPECTED_VAL_KEY:
            cur_parameter = list(self.cur_func.parameters.keys())[
                self.value_counter]
            return self.tokeniser.encode(cur_parameter)
        elif self.cur_state == JsonState.EXPECTED_VAL_VALUE:
            cur_parameter_value = list(self.cur_func.parameters.values())[
                self.value_counter - 1]
            return self.tokeniser.encode(cur_parameter_value.type)
        elif self.cur_state == JsonState

    def get_function_link(self, name: str) -> Functions:
        return next(fn for fn in self.func_list if fn.name == name)
