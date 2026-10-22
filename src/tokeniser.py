import regex
import json


class CustomQwenTokeniser:
    def __init__(self, path_to_vocab: str, merges_path: str) -> None:
        """Load the vocabulary and merge rules for a Qwen tokenizer.

        Args:
            path_to_vocab: Path to the tokenizer vocabulary JSON file.
            merges_path: Path to the byte-pair encoding merge rules.

        Raises:
            ValueError: If tokenizer data cannot be read or parsed.
        """
        self._pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)"\
            "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"\
            "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        self.path_to_vocab = path_to_vocab
        self.merges_path = merges_path
        self.token_to_id: dict[str, int] = self.json_to_dict()
        self.id_to_token: dict[int, str] = {
            id: token for token, id in self.token_to_id.items()}
        self.unicode_dict: dict[int, str] = self.bytes_to_unicode()
        self.decode_dict: dict[str, int] = {
            char: byte for byte, char in self.unicode_dict.items()}
        self.merges_dict: dict[tuple[str, str],
                               int] = self.create_merges_dict()

    def encode(self, text: str) -> list[int]:
        """Encode text into model token IDs.

        Args:
            text: Unicode text to tokenize.

        Returns:
            Token IDs produced by byte-pair encoding.
        """
        if not text:
            return []
        pre_token = regex.findall(self._pattern, text)
        raw_bytes = [byte.encode() for byte in pre_token]
        unicode_text = [[self.unicode_dict[byte]
                         for byte in key] for key in raw_bytes]
        best_index = None
        best_rank = None
        for word in unicode_text:
            while (True):
                for i in range(len(word) - 1):
                    if (word[i], word[i + 1]) in self.merges_dict:
                        if best_rank is None:
                            best_index = i
                            best_rank = self.merges_dict[word[i], word[i + 1]]
                        elif best_rank > self.merges_dict[
                            word[i], word[i + 1]
                        ]:
                            best_index = i
                            best_rank = self.merges_dict[word[i], word[i + 1]]
                if best_index is None:
                    break
                else:
                    word[best_index] = word[best_index] + word[best_index+1]
                    word.pop(best_index + 1)
                    best_index = None
                    best_rank = None
        id_list: list[int] = []
        for word in unicode_text:
            for token in word:
                id_list.append(self.token_to_id[token])
        return id_list

    def decode(self, id_list: list[int]) -> str:
        """Decode model token IDs into Unicode text.

        Args:
            id_list: Token IDs to decode.

        Returns:
            Decoded text, replacing malformed UTF-8 sequences.

        Raises:
            KeyError: If a token ID is absent from the vocabulary.
        """
        unicode_list = [self.id_to_token[id] for id in id_list]
        text_encoded = "".join(unicode_list)
        byte_values = (self.decode_dict[char] for char in text_encoded)
        raw_bytes = bytes(byte_values)
        result = raw_bytes.decode("utf-8", errors='replace')
        return (result)

    def json_to_dict(self) -> dict[str, int]:
        """Read the tokenizer vocabulary from JSON.

        Returns:
            A mapping from encoded token strings to token IDs.

        Raises:
            ValueError: If the vocabulary cannot be read or parsed.
        """
        try:
            with open(self.path_to_vocab) as vocab_file:
                data: dict[str, int] = json.load(vocab_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot load tokenizer vocabulary: {error}"
            ) from error
        return data

    def bytes_to_unicode(self) -> dict[int, str]:
        """Build the reversible byte-to-Unicode mapping used by GPT BPE.

        Returns:
            A mapping containing one printable Unicode value per byte.
        """
        bytes = list(list(range(ord("!"), ord("~") + 1)) +
                     list(range(ord("¡"), ord("¬") + 1)) +
                     list(range(ord("®"), ord("ÿ") + 1))
                     )
        n = 0
        chars = bytes[:]
        for i in range(256):
            if i not in bytes:
                bytes.append(i)
                chars.append(256 + n)
                n += 1
        characters = [chr(ch) for ch in chars]
        ret = {b: c for b, c in zip(bytes, characters)}
        return ret

    def create_merges_dict(self) -> dict[tuple[str, str], int]:
        """Load byte-pair merge ranks from the tokenizer merge file.

        Returns:
            Token-pair merge ranks keyed by the two component strings.

        Raises:
            ValueError: If the merge file cannot be read or parsed.
        """
        merge_dict: dict[tuple[str, str], int] = {}
        try:
            with open(self.merges_path) as fd:
                next(fd)
                for index, characters in enumerate(fd):
                    char_1, char_2 = characters.split()
                    merge_dict[char_1, char_2] = index
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Cannot load tokenizer merges: {error}"
            ) from error
        return merge_dict


def main() -> None:
    """Compare the custom tokenizer with the SDK on diagnostic inputs."""
    TEST_CASES: list[str] = [
        # 1. Граничные случаи пустых строк и пробелов
        "",
        " ",
        "   ",
        "Hello world",
        " hello world ",
        "hello   world",
        "\t",
        "\n",
        "\r\n",
        "\n\n\t\r\n",

        # 2. JSON-синтаксис, экранирование и спецсимволы (Function Calling)
        '{"name": "get_weather", "arguments": {"city": "Paris"}}',
        '{"name": "calculate", "arguments": {"a": 42, "b": 3.14}}',
        '{"parameters": {}}',
        "[]",
        '["item_1", "item_2", 123]',
        '{"key": "value with \\"quotes\\" and \\nnewlines"}',
        "::==>>",
        "---===***",
        "<!-- comment -->",

        # 3. Числа, плавающая точка, ведущие нули (проверка \p{N})
        "0",
        "1234567890",
        "0001",
        "42",
        "3.1415926535",
        "-100",
        "1e-5",
        "user_id_42",
        "item_v2_final_99",

        # 4. Английские сокращения и апострофы (ветка (?i:'s|'t|...))
        "I'm",
        "don't",
        "they'll",
        "it's",
        "we've",
        "you'd",
        "she're",
        "DON'T DO IT",
        "IT'S A TEST",
        "rock'n'roll",

        # 5. Идентификаторы, змеиный регистр и CamelCase
        "get_current_weather",
        "Small_LLM_Model",
        "__init__",
        "token_to_id",
        "API_KEY_SECRET",

        # 6. Кириллица и 2-байтовые последовательности UTF-8
        "Привет, мир!",
        "Тестирование токенизатора",
        "Функция вызова",
        "ёжик в тумане",
        "123 привет 456 мир",

        # 7. Расширенная латиница, акценты и диакритика
        "Bonjour le monde! Ça va?",
        "Schöne Grüße aus München",
        "¡Hola, señor!",
        "naïve café résumé",

        # 8. Эмодзи и 4-байтовые символы Unicode
        "🤖",
        "🚀",
        "🔥",
        "🤖🚀🔥",
        "Hello 👋 world 🌍!",
        "Python 🐍 is awesome ✨",

        # 9. Длинные слова, повторы и Out-of-Vocabulary (OOV)
        "supercalifragilisticexpialidocious",
        "antidisestablishmentarianism",
        "aaaaaaaaaaaaaaaaaaaa",
        "asdfghjklqwertyuiopzxcvbnm",
        "!@#$%^&*()_+{}[]:;\"'<>,.?/|\\~`",
    ]
    import llm_sdk

    model_class = getattr(llm_sdk, "Small_LLM_Model")
    model = model_class()
    path_to_vocab = model.get_path_to_vocab_file()
    merges_path = model.get_path_to_merges_file()
    tokeniser = CustomQwenTokeniser(path_to_vocab, merges_path)

    ids = tokeniser.encode("Hello World")
    tokeniser.decode(ids)

    for idx, test_case in enumerate(TEST_CASES):
        raw_expected = model.encode(test_case)
        if hasattr(raw_expected, "tolist"):
            expected = raw_expected.flatten().tolist()
        else:
            expected = list(raw_expected)
        actual = tokeniser.encode(test_case)

        if actual != expected:
            print(f"Wrong tokenising id: {idx}")
            print(expected)
            print(actual)

        actual_decode = tokeniser.decode(actual)
        expected_decode = model.decode(raw_expected)
        if actual_decode != expected_decode[0]:
            print(actual_decode, expected_decode[0])


if __name__ == "__main__":
    main()
