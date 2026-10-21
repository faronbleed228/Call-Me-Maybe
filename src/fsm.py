from __future__ import annotations

from collections import deque
from enum import Enum, auto
import json
import logging
import re
from typing import ClassVar, Protocol, Sequence

import numpy as np

from .tokeniser import CustomQwenTokeniser
from .trie import FunctionTrie, TokenTrieNode
from .validation import Functions


logger = logging.getLogger(__name__)


class LogitsModel(Protocol):
    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> Sequence[float]:
        """Return next-token logits for an encoded prompt.

        Args:
            input_ids: Complete encoded model context.

        Returns:
            One logit for every token in the model vocabulary.
        """
        ...


class JsonState(Enum):
    EXPECTED_NAME_KEY = auto()
    EXPECTED_FUNC_KEY = auto()
    EXPECTED_PARAM_KEY = auto()
    EXPECTED_VAL_KEY = auto()
    EXPECTED_VAL_VALUE = auto()
    EXPECTED_CLOSE = auto()


class DecisionState(Enum):
    DECISION = auto()
    PRE_DECIDED = auto()


class ValueType(Enum):
    STRING_TOKENS = auto()
    NUM_TOKENS = auto()
    BOOL_TOKENS = auto()


class FSMStack:
    MAX_VALUE_TOKENS = 128
    INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9]\d*)")
    NUMBER_PATTERN = re.compile(
        r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    )
    BOOLEAN_VALUES = ("true", "false")
    TOKEN_CACHE: ClassVar[
        dict[int, tuple[tuple[int, ...], ...]]
    ] = {}

    def __init__(
        self,
        trie: FunctionTrie,
        tokeniser: CustomQwenTokeniser,
    ) -> None:
        """Initialize constrained-decoding state and reusable token sets.

        Args:
            trie: Trie containing every permitted function name.
            tokeniser: Tokenizer used by the language model.

        Raises:
            ValueError: If a required JSON delimiter is not one token.
        """
        self.trie_root: TokenTrieNode = trie.root
        self.cur_trie: TokenTrieNode = self.trie_root
        self.cur_state = JsonState.EXPECTED_NAME_KEY
        self.tokeniser = tokeniser
        self.quote_token_id = self.single_token_id('"')
        self.comma_token_id = self.single_token_id(",")
        self.close_token_id = self.single_token_id("}")
        self.backslash_token_id = self.single_token_id("\\")
        (
            self.pre_number_tokens,
            self.pre_int_tokens,
            self.pre_str_tokens,
            self.pre_bool_tokens,
        ) = self.get_token_sets()
        self.cur_func: Functions

    def single_token_id(self, text: str) -> int:
        """Resolve text that must correspond to exactly one token.

        Args:
            text: Delimiter text to encode.

        Returns:
            The single encoded token ID.

        Raises:
            ValueError: If encoding produces zero or multiple tokens.
        """
        tokens = self.tokeniser.encode(text)
        if len(tokens) != 1:
            raise ValueError(f"Expected one token for {text!r}, got {tokens}")
        return tokens[0]

    def get_allowed_tokens(
        self,
        cur_value: int,
        cur_tokens: list[int],
    ) -> deque[int]:
        """Return tokens permitted by the current JSON and schema state.

        Args:
            cur_value: Index of the parameter currently being generated.
            cur_tokens: Tokens already generated for the current field.

        Returns:
            Token IDs that keep the partial output valid.

        Raises:
            RuntimeError: If the FSM is not in a model-decision state.
        """
        if self.cur_state == JsonState.EXPECTED_FUNC_KEY:
            allowed = deque(self.cur_trie.children)
            if self.cur_trie.func_name is not None:
                allowed.append(self.quote_token_id)
            return allowed
        if self.cur_state != JsonState.EXPECTED_VAL_VALUE:
            raise RuntimeError(f"No token decision for state {self.cur_state}")

        parameters = list(self.cur_func.parameters.items())
        value_name, parameter = parameters[cur_value]
        is_last = cur_value == len(parameters) - 1
        terminator = self.close_token_id if is_last else self.comma_token_id

        if parameter.type == "string":
            return self.string_tokens(cur_tokens, value_name == "regex")
        if parameter.type == "number":
            return self.number_tokens(cur_tokens, terminator)
        if parameter.type == "integer":
            return self.int_tokens(cur_tokens, terminator)
        return self.bool_tokens(cur_tokens)

    def string_tokens(
        self,
        cur_tokens: list[int],
        is_regex: bool = False,
    ) -> deque[int]:
        """Return tokens allowed inside a JSON string value.

        Args:
            cur_tokens: Tokens already generated for the value.
            is_regex: Require a compilable regular expression before closing.

        Returns:
            Permitted string-content and closing-quote token IDs.
        """
        current = self.tokeniser.decode(cur_tokens)
        can_finish = not is_regex or self.is_regex(current)
        if len(cur_tokens) >= self.MAX_VALUE_TOKENS and can_finish:
            return deque([self.quote_token_id])

        allowed = deque(self.pre_str_tokens)
        allowed.append(self.backslash_token_id)
        if can_finish:
            allowed.append(self.quote_token_id)
        return allowed

    @staticmethod
    def is_regex(pattern: str) -> bool:
        """Check whether text is a compilable regular expression.

        Args:
            pattern: Regular-expression source text.

        Returns:
            ``True`` when Python's regular-expression engine accepts it.
        """
        try:
            re.compile(pattern)
        except re.error:
            return False
        return True

    def number_tokens(
        self,
        cur_tokens: list[int],
        terminator: int,
    ) -> deque[int]:
        """Return tokens allowed for a general JSON number.

        Args:
            cur_tokens: Tokens already generated for the number.
            terminator: Structural token that ends this parameter.

        Returns:
            Permitted numeric and terminator token IDs.
        """
        return self.numeric_tokens(
            cur_tokens,
            terminator,
            self.pre_number_tokens,
            integer=False,
        )

    def int_tokens(
        self,
        cur_tokens: list[int],
        terminator: int,
    ) -> deque[int]:
        """Return tokens allowed for a JSON integer.

        Args:
            cur_tokens: Tokens already generated for the integer.
            terminator: Structural token that ends this parameter.

        Returns:
            Permitted integer and terminator token IDs.
        """
        return self.numeric_tokens(
            cur_tokens,
            terminator,
            self.pre_int_tokens,
            integer=True,
        )

    def numeric_tokens(
        self,
        cur_tokens: list[int],
        terminator: int,
        candidate_tokens: Sequence[int],
        integer: bool,
    ) -> deque[int]:
        """Filter candidate tokens against JSON numeric grammar.

        Args:
            cur_tokens: Tokens already generated for the number.
            terminator: Structural token that ends this parameter.
            candidate_tokens: Numeric vocabulary tokens to inspect.
            integer: Whether fractional and exponent syntax is forbidden.

        Returns:
            Tokens that preserve a valid numeric prefix.
        """
        current = self.tokeniser.decode(cur_tokens)
        complete = self.is_complete_number(current, integer)
        at_limit = len(cur_tokens) >= self.MAX_VALUE_TOKENS - 1
        allowed: deque[int] = deque()

        for token_id in candidate_tokens:
            candidate = self.tokeniser.decode(cur_tokens + [token_id])
            if not self.is_number_prefix(candidate, integer):
                continue
            if at_limit and not self.is_complete_number(candidate, integer):
                continue
            allowed.append(token_id)

        if complete:
            allowed.append(terminator)
        return allowed

    def bool_tokens(self, cur_tokens: list[int]) -> deque[int]:
        """Return tokens that preserve a ``true`` or ``false`` prefix.

        Args:
            cur_tokens: Tokens already generated for the Boolean value.

        Returns:
            Permitted Boolean token IDs.
        """
        current = self.tokeniser.decode(cur_tokens)
        allowed: deque[int] = deque()
        for token_id in self.pre_bool_tokens:
            candidate = current + self.tokeniser.decode([token_id])
            if any(
                value.startswith(candidate) for value in self.BOOLEAN_VALUES
            ):
                allowed.append(token_id)
        return allowed

    def get_token_sets(self) -> tuple[tuple[int, ...], ...]:
        """Classify vocabulary tokens for each supported parameter type.

        Returns:
            Cached number, integer, string, and Boolean token collections.
        """
        cache_key = id(self.tokeniser)
        cached = self.TOKEN_CACHE.get(cache_key)
        if cached is not None:
            return cached

        number_tokens: list[int] = []
        integer_tokens: list[int] = []
        string_tokens: list[int] = []
        boolean_tokens: list[int] = []
        for token_id in self.tokeniser.token_to_id.values():
            token = self.tokeniser.decode([token_id])
            if not token:
                continue
            if re.fullmatch(r"[0-9eE+.\-]+", token):
                number_tokens.append(token_id)
            if re.fullmatch(r"[0-9\-]+", token):
                integer_tokens.append(token_id)
            if '"' not in token and "\\" not in token:
                string_tokens.append(token_id)
            if any(token in value for value in self.BOOLEAN_VALUES):
                boolean_tokens.append(token_id)

        token_sets = (
            tuple(number_tokens),
            tuple(integer_tokens),
            tuple(string_tokens),
            tuple(boolean_tokens),
        )
        self.TOKEN_CACHE[cache_key] = token_sets
        return token_sets

    def is_complete_number(self, value: str, integer: bool) -> bool:
        """Check whether text is a complete JSON number.

        Args:
            value: Numeric text to validate.
            integer: Whether to require integer syntax.

        Returns:
            ``True`` when the entire value matches the requested grammar.
        """
        pattern = self.INTEGER_PATTERN if integer else self.NUMBER_PATTERN
        return pattern.fullmatch(value) is not None

    @staticmethod
    def is_number_prefix(value: str, integer: bool) -> bool:
        """Check whether text can still become a valid JSON number.

        Args:
            value: Partial numeric text to validate.
            integer: Whether fractional and exponent syntax is forbidden.

        Returns:
            ``True`` when further characters can complete the value.
        """
        if not value:
            return True

        index = 0
        if value[index] == "-":
            index += 1
            if index == len(value):
                return True

        if value[index] == "0":
            index += 1
            if index < len(value) and value[index].isdigit():
                return False
        elif value[index] in "123456789":
            while index < len(value) and value[index].isdigit():
                index += 1
        else:
            return False

        if integer or index == len(value):
            return index == len(value)

        if value[index] == ".":
            index += 1
            if index == len(value):
                return True
            if not value[index].isdigit():
                return False
            while index < len(value) and value[index].isdigit():
                index += 1
            if index == len(value):
                return True

        if index < len(value) and value[index] in "eE":
            index += 1
            if index == len(value):
                return True
            if value[index] in "+-":
                index += 1
                if index == len(value):
                    return True
            if not value[index].isdigit():
                return False
            while index < len(value) and value[index].isdigit():
                index += 1

        return index == len(value)


class FSM:
    MAX_GENERATED_TOKENS = 512
    COMPLETION_BIAS = 10.0
    REGEX_COMPLETION_BIAS = 20.0
    NUMERIC_COMPLETION_BIAS = 10.0

    def __init__(
        self,
        trie: FunctionTrie,
        tokeniser: CustomQwenTokeniser,
        prompt: str,
        llm_model: LogitsModel,
    ) -> None:
        """Initialize generation state for one model prompt.

        Args:
            trie: Trie restricting function-name generation.
            tokeniser: Tokenizer shared with the language model.
            prompt: Complete system and user prompt.
            llm_model: Model capable of returning next-token logits.
        """
        self.fsm_output_string: list[int] = []
        self.tokeniser = tokeniser
        self.model = llm_model
        self.fsm_stack = FSMStack(trie, tokeniser)
        self.cur_state = DecisionState.PRE_DECIDED
        self.prompt = self.init_system_prompt(prompt)
        self.returned_tokens: list[int] = []
        self.cur_func: Functions
        self.value_counter = 0
        self.value_type: ValueType
        self.value_dict = {
            "string": ValueType.STRING_TOKENS,
            "number": ValueType.NUM_TOKENS,
            "integer": ValueType.NUM_TOKENS,
            "boolean": ValueType.BOOL_TOKENS,
        }
        self.func_list = trie.function_list

    def main_loop(self) -> list[int]:
        """Generate one complete schema-constrained function call.

        Returns:
            Token IDs containing a valid function-call JSON object.

        Raises:
            RuntimeError: If generation exceeds its limit or reaches an
                invalid FSM state.
        """
        logger.debug("Main prompt: %s", self.tokeniser.decode(self.prompt))
        decisions = 0
        while True:
            if self.cur_state == DecisionState.PRE_DECIDED:
                if self.fsm_stack.cur_state == JsonState.EXPECTED_VAL_KEY:
                    self.set_value_type()
                self.pre_decided_tokens()
            else:
                decisions += 1
                if decisions > self.MAX_GENERATED_TOKENS:
                    raise RuntimeError("Generation exceeded the token limit")
                self.model_decide()

            self.transition()
            if self.fsm_stack.cur_state == JsonState.EXPECTED_NAME_KEY:
                return self.fsm_output_string

    def model_decide(self) -> None:
        """Choose and append the highest-logit currently allowed token.

        Raises:
            RuntimeError: If no token is valid or an allowed ID is outside
                the model vocabulary.
        """
        allowed = self.fsm_stack.get_allowed_tokens(
            self.value_counter,
            self.returned_tokens,
        )
        allowed_tokens = list(allowed)
        if not allowed_tokens:
            raise RuntimeError(
                f"No valid tokens for state {self.fsm_stack.cur_state}"
            )

        logger.debug("Allowed token count: %d", len(allowed_tokens))
        logits = np.asarray(
            self.model.get_logits_from_input_ids(self.prompt),
            dtype=np.float32,
        )
        allowed_logits = self.apply_logit_mask(logits, allowed_tokens)
        self.apply_completion_bias(allowed_logits)
        token_id = int(allowed_logits.argmax())
        logger.debug("Chosen token: %s", self.tokeniser.decode([token_id]))

        output_tokens: list[int] = []
        if not self.is_control_token(token_id):
            output_tokens = self.json_safe_tokens(token_id)
        self.returned_tokens.append(token_id)
        self.prompt.extend(output_tokens)
        self.fsm_output_string.extend(output_tokens)

    def is_control_token(self, token_id: int) -> bool:
        """Check whether a selected token only controls an FSM transition.

        Args:
            token_id: Selected model token ID.

        Returns:
            ``True`` when the token must not be copied to output directly.
        """
        if self.fsm_stack.cur_state == JsonState.EXPECTED_FUNC_KEY:
            return (
                token_id == self.fsm_stack.quote_token_id
                and self.fsm_stack.cur_trie.func_name is not None
            )
        if (
            self.fsm_stack.cur_state == JsonState.EXPECTED_VAL_VALUE
            and self.value_type == ValueType.NUM_TOKENS
        ):
            return token_id == self.numeric_terminator_id()
        return False

    def json_safe_tokens(self, token_id: int) -> list[int]:
        """Escape a generated token when it appears in a JSON string.

        Args:
            token_id: Raw model token ID.

        Returns:
            One or more token IDs containing JSON-safe text.
        """
        is_function_name = (
            self.fsm_stack.cur_state == JsonState.EXPECTED_FUNC_KEY
        )
        is_string_value = (
            self.fsm_stack.cur_state == JsonState.EXPECTED_VAL_VALUE
            and self.value_type == ValueType.STRING_TOKENS
            and token_id != self.fsm_stack.quote_token_id
        )
        if is_function_name or is_string_value:
            token = self.tokeniser.decode([token_id])
            escaped_token = json.dumps(token, ensure_ascii=False)[1:-1]
            if escaped_token != token:
                return self.tokeniser.encode(escaped_token)
        return [token_id]

    def pre_decided_tokens(self) -> None:
        """Append deterministic JSON structure for the current state.

        Raises:
            RuntimeError: If called from a state without fixed output.
        """
        state = self.fsm_stack.cur_state
        if state == JsonState.EXPECTED_NAME_KEY:
            text = '{"name":"'
        elif state == JsonState.EXPECTED_PARAM_KEY:
            text = '","parameters":{'
        elif state == JsonState.EXPECTED_VAL_KEY:
            parameter_name = list(self.cur_func.parameters)[self.value_counter]
            separator = "," if self.value_counter else ""
            key = json.dumps(parameter_name, ensure_ascii=False)
            opening_quote = (
                '"' if self.value_type == ValueType.STRING_TOKENS else ""
            )
            text = f"{separator}{key}:{opening_quote}"
        elif state == JsonState.EXPECTED_CLOSE:
            text = "}}"
        else:
            raise RuntimeError(f"No predetermined output for state {state}")

        tokens = self.tokeniser.encode(text)
        self.fsm_output_string.extend(tokens)
        self.prompt.extend(tokens)

    def set_value_type(self) -> None:
        """Set the internal value category for the current parameter."""
        parameter = list(self.cur_func.parameters.values())[self.value_counter]
        self.value_type = self.value_dict[parameter.type]
        logger.debug("Changed value type to %s", parameter.type)

    @staticmethod
    def apply_logit_mask(
        logits: np.ndarray,
        allowed_tokens: list[int],
    ) -> np.ndarray:
        """Mask logits for every token forbidden by the FSM.

        Args:
            logits: Unfiltered model logits for the vocabulary.
            allowed_tokens: Token IDs permitted in the current state.

        Returns:
            A copy containing negative infinity for forbidden tokens.

        Raises:
            RuntimeError: If an allowed token ID is outside the logits array.
        """
        if any(token < 0 or token >= logits.size for token in allowed_tokens):
            raise RuntimeError("Allowed token is outside the model vocabulary")
        logits_mask = np.full_like(logits, -np.inf, dtype=np.float32)
        logits_mask[allowed_tokens] = logits[allowed_tokens]
        return logits_mask

    def apply_completion_bias(self, logits: np.ndarray) -> None:
        """Increase terminator logits after a value becomes complete.

        Args:
            logits: Masked logits modified in place.
        """
        if (self.fsm_stack.cur_state != JsonState.EXPECTED_VAL_VALUE
                or not self.returned_tokens):
            return

        if self.value_type == ValueType.STRING_TOKENS:
            quote_token = self.fsm_stack.quote_token_id
            if np.isfinite(logits[quote_token]):
                parameters = list(self.cur_func.parameters)
                parameter_name = parameters[self.value_counter]
                bias = self.COMPLETION_BIAS
                if parameter_name == "regex":
                    bias = self.REGEX_COMPLETION_BIAS
                logits[quote_token] += bias
            return

        if self.value_type == ValueType.NUM_TOKENS:
            numeric_parameters = list(self.cur_func.parameters.values())
            parameter = numeric_parameters[self.value_counter]
            value = self.tokeniser.decode(self.returned_tokens)
            is_fraction = any(character in value for character in ".eE")
            is_complete = self.fsm_stack.is_complete_number(value, False)
            terminator = self.numeric_terminator_id()
            if (parameter.type == "number" and is_fraction and is_complete
                    and np.isfinite(logits[terminator])):
                logits[terminator] += self.NUMERIC_COMPLETION_BIAS

    def transition(self) -> None:
        """Advance the FSM after appending deterministic or model output."""
        state = self.fsm_stack.cur_state
        logger.debug("Current state: %s", state)

        if state == JsonState.EXPECTED_NAME_KEY:
            self.fsm_stack.cur_state = JsonState.EXPECTED_FUNC_KEY
            self.cur_state = DecisionState.DECISION
        elif state == JsonState.EXPECTED_FUNC_KEY:
            self.transition_function_name()
        elif state == JsonState.EXPECTED_PARAM_KEY:
            if self.cur_func.parameters:
                self.fsm_stack.cur_state = JsonState.EXPECTED_VAL_KEY
            else:
                self.fsm_stack.cur_state = JsonState.EXPECTED_CLOSE
        elif state == JsonState.EXPECTED_VAL_KEY:
            self.returned_tokens.clear()
            self.fsm_stack.cur_state = JsonState.EXPECTED_VAL_VALUE
            self.cur_state = DecisionState.DECISION
        elif state == JsonState.EXPECTED_VAL_VALUE:
            if self.value_is_complete():
                self.finish_value()
        elif state == JsonState.EXPECTED_CLOSE:
            self.reset_prompt()

    def transition_function_name(self) -> None:
        """Advance through the function trie after one selected token.

        Raises:
            RuntimeError: If the selected token is absent from the trie.
        """
        token_id = self.returned_tokens[-1]
        if token_id == self.fsm_stack.quote_token_id:
            self.finish_function_name()
            return

        try:
            self.fsm_stack.cur_trie = self.fsm_stack.cur_trie.children[
                token_id
            ]
        except KeyError as error:
            raise RuntimeError("Invalid function-name token") from error

        current = self.fsm_stack.cur_trie
        if current.func_name is not None and not current.children:
            self.finish_function_name()

    def finish_function_name(self) -> None:
        """Bind a completed trie path to its function schema.

        Raises:
            RuntimeError: If the current trie node has no function name or the
                name does not correspond to an available schema.
        """
        function_name = self.fsm_stack.cur_trie.func_name
        if function_name is None:
            raise RuntimeError("Incomplete function name")
        logger.debug("Extracted function name: %s", function_name)
        self.cur_func = self.get_function_link(function_name)
        self.fsm_stack.cur_func = self.cur_func
        self.fsm_stack.cur_state = JsonState.EXPECTED_PARAM_KEY
        self.cur_state = DecisionState.PRE_DECIDED
        self.returned_tokens.clear()

    def value_is_complete(self) -> bool:
        """Check whether the current parameter has reached its terminator.

        Returns:
            ``True`` when generation for the current value is complete.
        """
        if self.value_type == ValueType.STRING_TOKENS:
            return self.returned_tokens[-1] == self.fsm_stack.quote_token_id
        if self.value_type == ValueType.NUM_TOKENS:
            return self.returned_tokens[-1] == self.numeric_terminator_id()
        value = self.tokeniser.decode(self.returned_tokens)
        return value in self.fsm_stack.BOOLEAN_VALUES

    def finish_value(self) -> None:
        """Move generation to the next parameter or closing JSON state."""
        self.value_counter += 1
        if self.value_counter == len(self.cur_func.parameters):
            self.fsm_stack.cur_state = JsonState.EXPECTED_CLOSE
        else:
            self.fsm_stack.cur_state = JsonState.EXPECTED_VAL_KEY
        self.cur_state = DecisionState.PRE_DECIDED

    def numeric_terminator_id(self) -> int:
        """Return the structural token that follows the current number.

        Returns:
            A comma token for intermediate parameters or a closing-brace token
            for the last parameter.
        """
        is_last = self.value_counter == len(self.cur_func.parameters) - 1
        if is_last:
            return self.fsm_stack.close_token_id
        return self.fsm_stack.comma_token_id

    def reset_prompt(self) -> None:
        """Reset mutable FSM state after completing one function call."""
        self.cur_state = DecisionState.PRE_DECIDED
        self.fsm_stack.cur_state = JsonState.EXPECTED_NAME_KEY
        self.fsm_stack.cur_trie = self.fsm_stack.trie_root
        self.value_counter = 0
        self.returned_tokens.clear()

    def init_system_prompt(self, prompt: str) -> list[int]:
        """Encode a new model prompt and clear prior output.

        Args:
            prompt: Complete prompt text to encode.

        Returns:
            Encoded prompt token IDs.
        """
        self.fsm_output_string.clear()
        return self.tokeniser.encode(prompt)

    def get_function_link(self, name: str) -> Functions:
        """Find a validated function schema by name.

        Args:
            name: Function name selected through the trie.

        Returns:
            Matching validated function definition.

        Raises:
            RuntimeError: If no available function has the requested name.
        """
        try:
            return next(
                function for function in self.func_list
                if function.name == name
            )
        except StopIteration as error:
            raise RuntimeError(f"Unknown function selected: {name}") from error
