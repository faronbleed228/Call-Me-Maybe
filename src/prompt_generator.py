from .validation import Prompt, Functions
from pathlib import Path
import json


PROMPT_STR = """"
    YO iam {user}

    IAM {years}


"""


class PromptGenerator():
    def __init__(self, functions: list[Functions]) -> None:
        self.functions = functions
        self.prompt_template: str = ("<|im_start|>system\n"
                                     "You are a helpful assistant with "
                                     "access to the following functions:\n"
                                     "{functions}\n"
                                     "To call a function, respond strictly with a "
                                     "JSON object in this format:\n"
                                     '{{"name": "<function_name>", "parameters": {{<arguments>}}}}'
                                     "<|im_end|>\n<|im_start|>user\n"
                                     "{prompt}\n<|im_end|>"
                                     "<|im_start|>assistant\n"
                                     "{{")
        self.functions_description: str = self.get_descriptions()

    def get_descriptions(self) -> str:
        functions_list: list[str] = ["["]
        for function in self.functions:
            dump_function = function.model_dump(
                exclude_none=True, exclude={"returns"})
            readable_json = json.dumps(dump_function, indent=2)
            functions_list.append(f"{readable_json}")
        functions_list.append("]")
        return ",\n".join(functions_list)

    def build_gen_promt(self, prompt: str) -> str:
        return self.prompt_template.format(
            functions=self.functions_description, prompt=prompt)


if __name__ == "__main__":
    print(PROMPT_STR.format(user="lol", years=19))
