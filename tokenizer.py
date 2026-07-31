import regex
import llm_sdk


class CastomQwenTokeniser():
    def __init__(self, vocab_path: str, merges_path: str):
        self._vocab_path = vocab_path
        self._merges_path = merges_path
        self._pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)"\
            "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"\
            "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        self._unicode_dict = self.bytes_to_unicode()
        self._merges_dict = self.merge_dict()

    def encode(self, text: str) -> list[int]:
        pre_token = regex.findall(self._pattern, text)
        raw_bytes = [byte.encode() for byte in pre_token]
        unicode_ch = [[self._unicode_dict[char]
                      for char in key] for key in raw_bytes]
        i = 0
        for word in unicode_ch:
            while (True):
                i = 0
                best_pos = None
                best_rank = None
                for i in range(len(word) - 1):
                    if (word[i], word[i + 1]) in self._merges_dict:
                        if best_rank is None:
                            best_pos = i
                            best_rank = self._merges_dict[word[i], word[i + 1]]
                        elif best_rank > self._merges_dict[word[i], word[i + 1]]:
                            best_pos = i
                            best_rank = self._merges_dict[word[i], word[i + 1]]
                if best_pos is None:
                    break
                else:
                    word[best_pos] = word[best_pos] + word[best_pos + 1]
                    word.pop(best_pos + 1)
        print(unicode_ch)

    def decode(self, ids: list[int]) -> str:
        pass

    def bytes_to_unicode(self) -> dict[int, str]:
        bs = list(list(range(ord("!"), ord("~") + 1)) +
                  list(range(ord("¡"), ord("¬") + 1)) +
                  list(range(ord("®"), ord("ÿ") + 1)))
        ch = bs[:]
        i = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                ch.append(256 + i)
                i += 1
        ch = [chr(c) for c in ch]
        return dict(zip(bs, ch))

    def merge_dict(self) -> dict[tuple[str, str], int]:
        ret = {}
        with open(self._merges_path) as f:
            next(f)
            for rank, line in enumerate(f):
                line = line.strip("\n")
                token_0, token_01 = line.split()
                ret[(token_0, token_01)] = rank
        return ret

    def vocab_dict():
        pass


def main():
    LLM_MODEL = llm_sdk.Small_LLM_Model()
    test_token = CastomQwenTokeniser(
        LLM_MODEL.get_path_to_vocab_file(), LLM_MODEL.get_path_to_merges_file())
    test_token.encode("Hello World.")


def test():
    rules = {("h", "e"): 0, ("he", "l"): 1, ("p", "0"): 2}
    word = ["h", "e", "l", "o", "p", "0"]
    while True:
        best_rank = None
        best_pos = None
        for i in range(len(word) - 1):
            if (word[i], word[i + 1]) in rules:
                if best_rank is None:
                    best_rank = rules[word[i], word[i + 1]]
                    best_pos = i
                elif best_rank > rules[word[i], word[i + 1]]:
                    best_rank = rules[word[i], word[i + 1]]
                    best_pos = i
        if best_pos is None:
            break
        else:
            word[best_pos] = word[best_pos] + word[best_pos + 1]
            word.pop(best_pos + 1)

    print(word)


if __name__ == "__main__":
    main()
