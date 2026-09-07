import numpy as np
from enum import Enum, auto


class MocKQwenTokeniser():
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "system": 151646,
            "assistant": 151647,
            "user": 151648,
            '{"name":': 1001,
            ' "fn_add_numbers"': 1002,
            ', "parameters": {': 1003,
            '"a":': 1004,
            '"b":': 1005,
            "}": 1006,
            "}}": 1007
        }
        self.inverse_vocab: dict[int, str] = {
            value: key for key, value in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        text_list = text.split()
        for word in text_list:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(999999)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.inverse_vocab.get(token, "<unk>") for token in tokens])


class MockQwenModel():

    def __init__(self, vocab_size: int = 151648) -> None:
        self.vocab_size = vocab_size

    def gey_logis_from_ids(self, input_ids: list[int]) -> np.ndarray:
        return np.random.uniform(-5.0, 5.0, size=(self.vocab_size)).astype(np.float32)


class JsonState(Enum):
    START = auto()
    EXPECTED_NAME_KEY = auto()
    EXPECTED_FUNC_KEY = auto()
    EXPECTED_PARAM_KEY = auto()
    EXPECTED_VAL_KEY = auto()
    EXPECTED_VAL_VALUE = auto()
    EXPECTED_CLOESE = auto()


class MockFSM():
    def __init__(self) -> None:
        self.cur_state = JsonState.START

    def transition(self) -> None:
        pass

    def get_allowed_tokens(self) -> list[int]:
        allowed_tokens: list[int] = []
        if self.cur_state == JsonState.Start:
            allowed_tokens.append[]


def main():
    mock_tokeniser = MocKQwenTokeniser()
    mock_model = MockQwenModel()
    input_ids = mock_tokeniser.encode("<|im_start|> system assistant user")
    print(input_ids)
    logits = mock_model.gey_logis_from_ids(input_ids)
    print(logits[151645])


if __name__ == "__main__":
    main()
