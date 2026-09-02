from .validation import Prompt, Functions
from pathlib import Path
from typing import Callable


PROMPT_STR = """"
    YO iam {user}

    IAM {years}


"""


class PromptGenerator():
    def __init__(self, functions: list[Functions]) -> None:
        self.functions = functions
        self.prompt_dict: dict[str, str] = {
            "function_name": ("<|im_start|>system\n"
                              "You are a helpful assistant with "
                              "access to the following functions:\n"
                              "{functions}\nRespond with function name only.\n"
                              "<|im_end|>\n<|im_start|>user\n"
                              "{prompt}\n<|im_end|>")
        }
        self.promt_functions: dict[str, Callable] = {
            "function_name_prompt": self.function_name_prompt
        }

    def prompt_generator(self, prompt_type: str, prompt: str):
        pass

    def function_name_prompt(self) -> str:
        final_prompt =


if __name__ == "__main__":
    print(PROMPT_STR.format(user="lol", years=19))
