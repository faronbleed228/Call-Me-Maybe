import llm_sdk
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Sequence
from .validation import validate_files, ValError
from .trie import FunctionTrie
from .tokeniser import CustomQwenTokeniser
from .prompt_generator import PromptGenerator
from sys import stderr
import pydantic as pd


def main() -> int:
    # model = llm_sdk.Small_LLM_Model()
    # path_to_vocab = model.get_path_to_vocab_file()
    # merges_path = model.get_path_to_merges_file()
    # tokeniser = CustomQwenTokeniser(path_to_vocab, merges_path)

    args = argument_pasrser()
    try:
        functions, prompts = validate_files(
            args.functions_definition, args.input, args.output)
    except ValError as e:
        print(e, file=stderr)
        return 1
    # testtrie = FunctionTrie(functions)
    # testtrie.build(tokeniser)
    prompt_generator = PromptGenerator(functions)
    print(prompt_generator.build_gen_promt(prompts[0].prompt))

    return 0


def argument_pasrser(argv: Sequence[str] | None = None) -> Namespace:
    pasrser = ArgumentParser(
        "call-me-maybe", formatter_class=ArgumentDefaultsHelpFormatter)
    pasrser.add_argument(
        "--functions_definition", help="Defines path to function definitions",
        default=Path("data/data/input/functions_definition.json"),
        type=Path)
    pasrser.add_argument("--input", help="Defines path to input file",
                         default=Path(
                             "data/data/input/function_calling_tests.json"),
                         type=Path)
    pasrser.add_argument("--output", help="Defines path to oputput file",
                         default=Path("data/output/function_calls.json"),
                         type=Path)

    args: Namespace = pasrser.parse_args(argv)
    return args
