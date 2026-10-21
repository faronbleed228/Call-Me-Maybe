from __future__ import annotations
from dataclasses import dataclass, field
from .validation import Functions
from .tokeniser import CustomQwenTokeniser


@dataclass
class TokenTrieNode:
    children: dict[int, TokenTrieNode] = field(default_factory=dict)
    func_name: str | None = None


class FunctionTrie:
    def __init__(self, function_list: list[Functions]) -> None:
        """Initialize an empty trie for the available functions.

        Args:
            function_list: Validated function definitions to index.
        """
        self.root = TokenTrieNode()
        self.function_list = function_list

    def build(self, tokeniser: CustomQwenTokeniser) -> None:
        """Populate the trie with tokenized function names.

        Args:
            tokeniser: Tokenizer used by the language model.
        """
        for function in self.function_list:
            name = function.name
            function_ids = tokeniser.encode(name)
            self.insert(function_ids, name)

    def insert(self, function_ids: list[int],
               func_name: str) -> None:
        """Insert one tokenized function name into the trie.

        Args:
            function_ids: Token IDs forming the function name.
            func_name: Original function name stored at the terminal node.
        """
        current = self.root
        for id in function_ids:
            if id in current.children:
                current = current.children[id]
            else:
                new_node = TokenTrieNode()
                current.children[id] = new_node
                current = new_node
        current.func_name = func_name
