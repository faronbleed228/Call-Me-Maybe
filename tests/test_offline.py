"""Offline decoder checks using tiny vocabulary and scripted logits."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pydantic import ValidationError

from src.fsm import FSM, FSMStack, JsonState, ValueType
from src.tokeniser import CustomQwenTokeniser
from src.trie import FunctionTrie
from src.validation import Functions, Prompt, ValError, validate_files


class TinyTokenizer(CustomQwenTokeniser):
    def __init__(self) -> None:
        self.token_to_id = {chr(i): i for i in range(128)}

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, id_list: list[int]) -> str:
        return "".join(chr(i) for i in id_list)


class ScriptedModel:
    def __init__(self, decisions: str) -> None:
        self.decisions = iter(decisions)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [-1000.0] * 128
        logits[ord(next(self.decisions))] = 1000.0
        return logits


def function(name: str, parameters: dict[str, dict[str, str]]) -> Functions:
    return Functions.model_validate({
        "name": name, "description": "", "parameters": parameters,
        "returns": {"type": "string"}})


class OfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        FSMStack.TOKEN_CACHE.clear()
        self.tokenizer = TinyTokenizer()

    def decoder(self, functions: list[Functions], decisions: str) -> FSM:
        trie = FunctionTrie(functions)
        trie.build(self.tokenizer)
        return FSM(trie, self.tokenizer, "test", ScriptedModel(decisions))

    def test_number_prefixes(self) -> None:
        for value in ("", "-", "0", "-0", "1.", "1e", "1e-", "1e-3"):
            self.assertTrue(FSMStack.is_number_prefix(value, False), value)
        for value in ("00", "+1", ".", "1e--", "1.e", "1e."):
            self.assertFalse(FSMStack.is_number_prefix(value, False), value)
        for value in ("1.", "1e2", "00", "-."):
            self.assertFalse(FSMStack.is_number_prefix(value, True), value)

    def test_integer_float_boolean_generation(self) -> None:
        fsm = self.decoder([function("f", {
            "i": {"type": "integer"}, "n": {"type": "number"},
            "b": {"type": "boolean"}})], "f-42,1.25e-2,true")
        result = json.loads(self.tokenizer.decode(fsm.main_loop()))
        self.assertEqual(result, {"name": "f", "parameters": {
            "i": -42, "n": 0.0125, "b": True}})

    def test_trie_prefix_and_empty_parameters(self) -> None:
        functions = [function("f", {}), function("foo", {})]
        for decisions, expected in [('f"', "f"), ("foo", "foo")]:
            fsm = self.decoder(functions, decisions)
            result = json.loads(self.tokenizer.decode(fsm.main_loop()))
            self.assertEqual(result, {"name": expected, "parameters": {}})

    def test_escaping_names_keys_and_values(self) -> None:
        fsm = self.decoder([function('f"\\', {'k"\\': {"type": "string"}})],
                           'f"\\a\n\t\\"')
        result = json.loads(self.tokenizer.decode(fsm.main_loop()))
        self.assertEqual(result, {"name": 'f"\\',
                                  "parameters": {'k"\\': 'a\n\t\\'}})
        fsm.fsm_stack.cur_state = JsonState.EXPECTED_VAL_VALUE
        fsm.value_type = ValueType.STRING_TOKENS
        for char in ("\n", "\t", "\\", "\x00"):
            escaped = self.tokenizer.decode(fsm.json_safe_tokens(ord(char)))
            self.assertEqual(json.loads('"' + escaped + '"'), char)

    def test_limits_and_logit_mask(self) -> None:
        fsm = self.decoder([function("f", {})], "f")
        fsm.MAX_GENERATED_TOKENS = 0
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            fsm.main_loop()
        stack = fsm.fsm_stack
        allowed = stack.string_tokens([ord("a")] * stack.MAX_VALUE_TOKENS)
        self.assertEqual(list(allowed), [ord('"')])
        logits = np.array([1, 2, 3], dtype=np.float32)
        masked = FSM.apply_logit_mask(logits, [1])
        self.assertTrue(np.isneginf(masked[0]))
        self.assertEqual(masked[1], 2)
        self.assertEqual(logits[0], 1)
        with self.assertRaises(RuntimeError):
            FSM.apply_logit_mask(logits, [3])

    def test_validation(self) -> None:
        for name in ("", "   "):
            with self.assertRaises(ValidationError):
                function(name, {})
        with self.assertRaises(ValidationError):
            Prompt(prompt=" ")
        with self.assertRaises(ValidationError):
            function("f", {"x": {"type": "object"}})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            functions = root / "functions.json"
            prompts = root / "prompts.json"
            prompts.write_text('[{"prompt":"test"}]', encoding="utf-8")
            f = function("f", {}).model_dump()
            functions.write_text(json.dumps([f, f]), encoding="utf-8")
            with self.assertRaisesRegex(ValError, "unique"):
                validate_files(functions, prompts, root / "out.json")
            for invalid in ("[]", "{broken", "{}"):
                functions.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValError):
                    validate_files(functions, prompts, root / "out.json")
            functions.write_text(json.dumps([f]), encoding="utf-8")
            validated, requests = validate_files(
                functions, prompts, root / "output" / "calls.json")
            self.assertEqual(len(validated), 1)
            self.assertEqual(requests[0].prompt, "test")
