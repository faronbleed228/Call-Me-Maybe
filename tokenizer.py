class CastomQwenTokeniser():
    def __init__(self, vocab_path: str, merges_path: str):
        self._vocab = vocab_path
        self._merges = merges_path

    def encode(self, text: str) -> list[int]:
        utf_8_text = text.encode()
        utf_8_text.b

    def decode(self, ids: list[int]) -> str:
        pass

    def bytes_to_unicode(raw_bytes: list[int]) -> dict[int:str]:
        bs = list(ord(range("!", "~" + 1)),
                  ord(range("¡", "¬" + 1)), ord(range("®", "ÿ")))
        cs = bs

        i = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + i)
                i += 1
        cs = [chr(x) for x in cs]
        return dict(zip(bs, cs))
