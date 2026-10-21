import json
from pathlib import Path
import re
from typing import Any

from .tokeniser import CustomQwenTokeniser
from .validation import Functions


class JsonGenerator:
    def __init__(
        self,
        tokeniser: CustomQwenTokeniser,
        functions: list[Functions],
    ) -> None:
        """Initialize the result builder.

        Args:
            tokeniser: Tokenizer used to decode FSM output.
            functions: Function schemas used to normalize parameters.
        """
        self.tokeniser = tokeniser
        self.output_json: list[dict[str, Any]] = []
        self.functions = {function.name: function for function in functions}

    def main_loop(self, fsm_tokens: list[int], prompt: str) -> None:
        """Decode and append one generated function call.

        Args:
            fsm_tokens: Token IDs containing one complete JSON object.
            prompt: Original user request associated with the result.

        Raises:
            json.JSONDecodeError: If the generated tokens are not valid JSON.
            KeyError: If required output fields or schemas are missing.
        """
        result_entry: dict[str, Any] = {}
        fsm_string = self.tokeniser.decode(fsm_tokens)
        fsm_dict = json.loads(fsm_string)
        result_entry["prompt"] = prompt
        result_entry["name"] = fsm_dict["name"]
        result_entry["parameters"] = self.normalize_parameters(
            fsm_dict["name"],
            fsm_dict["parameters"],
            prompt,
        )
        self.output_json.append(result_entry)

    def normalize_parameters(
        self,
        function_name: str,
        parameters: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Normalize generated parameters according to their schema.

        Args:
            function_name: Name of the selected function.
            parameters: Parameters decoded from generated JSON.
            prompt: Original user request used for string recovery.

        Returns:
            Parameters with schema-compatible numbers and recovered strings.

        Raises:
            KeyError: If the function or one of its parameters is unknown.
            TypeError: If a numeric parameter cannot be converted to a float.
            ValueError: If a numeric parameter cannot be converted to a float.
        """
        function = self.functions[function_name]
        normalized: dict[str, Any] = {}
        only_one_parameter = len(function.parameters) == 1

        for name, definition in function.parameters.items():
            value = parameters[name]
            if definition.type == "number":
                normalized[name] = float(value)
            elif definition.type == "string":
                normalized[name] = self.recover_source_string(
                    str(value),
                    name,
                    prompt,
                    only_one_parameter,
                )
            else:
                normalized[name] = value
        return normalized

    @staticmethod
    def recover_source_string(
        value: str,
        parameter_name: str,
        prompt: str,
        only_one_parameter: bool,
    ) -> str:
        """Recover exact source text when the model slightly alters a string.

        Args:
            value: String produced by constrained decoding.
            parameter_name: Schema name of the string parameter.
            prompt: Original natural-language request.
            only_one_parameter: Whether the selected function has one argument.

        Returns:
            Exact prompt text when a reliable match exists. Otherwise, the
            original ``value``.
        """
        if only_one_parameter:
            readable_name = re.escape(parameter_name.replace("_", " "))
            explicit_value = re.search(
                rf"(?:^|\b){readable_name}\s*:\s*(.+)$",
                prompt,
                flags=re.IGNORECASE,
            )
            if explicit_value is not None:
                return explicit_value.group(1).strip()

        if value in prompt:
            return value

        slash_normalized_prompt = prompt.replace("\\", "/")
        slash_normalized_value = value.replace("\\", "/")
        value_position = slash_normalized_prompt.find(slash_normalized_value)
        if value_position >= 0:
            value_end = value_position + len(slash_normalized_value)
            return prompt[value_position:value_end]

        quoted_values: list[tuple[str, str]] = re.findall(
            r"(['\"])(.*?)\1", prompt
        )
        matching_values = [
            candidate
            for _, candidate in quoted_values
            if value.startswith(candidate)
        ]
        if matching_values:
            return max(matching_values, key=len)
        return value

    def json_dump(self, file_name: Path) -> None:
        """Write all accumulated function calls to a JSON file.

        Args:
            file_name: Destination JSON path.

        Raises:
            OSError: If the destination cannot be written.
        """
        with open(file_name, "w", encoding="utf-8") as fd:
            json.dump(self.output_json, fd, indent=2, ensure_ascii=False)
