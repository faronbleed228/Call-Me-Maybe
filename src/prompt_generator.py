import json

from .validation import Functions


class PromptGenerator:
    def __init__(self, functions: list[Functions]) -> None:
        """Initialize the generator with available function definitions.

        Args:
            functions: Functions exposed to the language model.
        """
        self.functions = functions
        self.prompt_template = (
            "<|im_start|>system\n"
            "You are a helpful assistant with access to the following "
            "functions:\n"
            "{functions}\n"
            "To call a function, respond strictly with a JSON object in "
            "this format:\n"
            '{{"name": "<function_name>", "parameters": '
            "{{<arguments>}}}}"
            "<|im_end|>\n<|im_start|>user\n"
            "{prompt}\n<|im_end|>"
            "<|im_start|>assistant\n"
        )
        self.functions_description: str = self.get_descriptions()

    def get_descriptions(self) -> str:
        """Serialize function metadata for inclusion in the system prompt.

        Returns:
            A JSON string containing names, descriptions, and parameters.
        """
        descriptions = [
            function.model_dump(exclude_none=True, exclude={"returns"})
            for function in self.functions
        ]
        return json.dumps(descriptions, indent=2, ensure_ascii=False)

    def build_gen_promt(self, prompt: str) -> str:
        """Build the complete chat prompt for one user request.

        Args:
            prompt: Natural-language function call request.

        Returns:
            The system, user, and assistant prompt in the model chat format.
        """
        return self.prompt_template.format(
            functions=self.functions_description, prompt=prompt)
