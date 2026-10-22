from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import json
import logging
from pathlib import Path
from sys import stderr
from typing import Sequence

from .fsm import FSM
from .json_generator import JsonGenerator
from .prompt_generator import PromptGenerator
from .tokeniser import CustomQwenTokeniser
from .trie import FunctionTrie
from .validation import validate_files, ValError


file_handler = logging.FileHandler("app.log", mode="w", encoding="utf-8")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[file_handler],
)


def main() -> int:
    """Run constrained function-call generation from command-line inputs.

    Returns:
        Zero on success and one when validation or generation fails.
    """
    args = argument_parser()
    try:
        functions, prompts = validate_files(
            args.functions_definition,
            args.input,
            args.output,
        )
        import llm_sdk

        print("Loading model...", flush=True)
        model_class = getattr(llm_sdk, "Small_LLM_Model")
        model = model_class()
        path_to_vocab = model.get_path_to_vocab_file()
        merges_path = model.get_path_to_merges_file()
        tokeniser = CustomQwenTokeniser(path_to_vocab, merges_path)

        function_trie = FunctionTrie(functions)
        function_trie.build(tokeniser)
        prompt_generator = PromptGenerator(functions)
        json_generator = JsonGenerator(tokeniser, functions)

        prompt_count = len(prompts)
        print(f"Processing {prompt_count} prompts...", flush=True)
        for index, prompt_entry in enumerate(prompts, start=1):
            prompt = prompt_entry.prompt
            print(f"[{index}/{prompt_count}] {prompt}", flush=True)
            master_prompt = prompt_generator.build_gen_promt(prompt)
            fsm = FSM(function_trie, tokeniser, master_prompt, model)
            fsm_tokens = fsm.main_loop()
            json_generator.main_loop(fsm_tokens, prompt)
            print(f"[{index}/{prompt_count}] done", flush=True)

        json_generator.json_dump(args.output)
        print(f"Results written to {args.output}", flush=True)
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        ValError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=stderr)
        return 1
    return 0


def argument_parser(argv: Sequence[str] | None = None) -> Namespace:
    """Parse command-line paths used by the application.

    Args:
        argv: Optional argument sequence, or ``None`` to use ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = ArgumentParser(
        "call-me-maybe", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--functions_definition", help="Defines path to function definitions",
        default=Path("data/input/functions_definition.json"),
        type=Path)
    parser.add_argument(
        "--input",
        help="Defines path to input file",
        default=Path("data/input/function_calling_tests.json"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        help="Defines path to output file",
        default=Path("data/output/function_calling_results.json"),
        type=Path,
    )

    args: Namespace = parser.parse_args(argv)
    return args
