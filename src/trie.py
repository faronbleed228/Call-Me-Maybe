from __future__ import annotations
from dataclasses import dataclass, field
from .validation import Functions
from .tokeniser import CustomQwenTokeniser


@dataclass
class TokenTrieNode:
    children: dict[int, TokenTrieNode] = field(default_factory=dict)
    func_name: str | None = None


class FunctionTrie():
    def __init__(self, function_list: list[Functions]) -> None:
        self.root = TokenTrieNode()
        self.function_list = function_list

    def build(self, tokeniser: CustomQwenTokeniser) -> None:
        for function in self.function_list:
            name = function.name
            function_ids = tokeniser.encode(name)
            self.insert(function_ids, name)

    def insert(self, function_ids: list[int],
               func_name: str) -> None:
        current = self.root
        for id in function_ids:
            if id in current.children:
                current = current.children[id]
            else:
                new_node = TokenTrieNode()
                current.children[id] = new_node
                current = new_node
        current.func_name = func_name
