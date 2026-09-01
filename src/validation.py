from __future__ import annotations
from pydantic import BaseModel, ValidationError, TypeAdapter
from pathlib import Path
from typing import Literal, Union
from os import access, W_OK


class ValError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Functions(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: Union[ReturnType, list[ReturnType]]


class ParameterType(BaseModel):
    type: Literal["number", "string"]


class ReturnType(BaseModel):
    type: Literal["number", "string", "boolean"]


class Prompt(BaseModel):
    prompt: str


def validate_files(functions_file: Path, prompt_file: Path,
                   output_file: Path) -> tuple[list[Functions], list[Prompt]]:
    functions_adapter = TypeAdapter(list[Functions])
    prompt_adapter = TypeAdapter(list[Prompt])

    if str(output_file).endswith(("/", "\\")) or output_file.is_dir():
        raise ValError("Output file has to lead to file not directory")

    try:
        raw_functions = functions_file.read_text(encoding="utf-8")
        raw_prompts = prompt_file.read_text(encoding="utf-8")
        functions = functions_adapter.validate_json(raw_functions)
        prompts = prompt_adapter.validate_json(raw_prompts)
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        target = f"'{e.filename}'" if e.filename else ""
        raise ValError(f"Error: {e.strerror} {target}")

    except ValidationError as e:
        first_error = e.errors()[0]
        msg = first_error["msg"]
        path = ".".join(str(item) for item in first_error["loc"])
        raise ValError(
            f"Pydantic Validation error: path: {path}, message: {msg}")

    except ValueError as e:
        raise ValError(f"Error: {e}")

    if output_file.exists():
        if access(output_file, W_OK) is False:
            raise ValError("Permission denied for output path")
    elif access(output_dir, W_OK) is False:
        raise ValError("Permission denied for output path")

    if not functions:
        raise ValError("Functions definition file cannot be empty")
    if not prompts:
        raise ValError("Prompt tests file cannot be empty")

    return functions, prompts
