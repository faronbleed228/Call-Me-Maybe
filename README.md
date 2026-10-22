*This project has been created as part of the 42 curriculum by ypopovyc.*

# Call Me Maybe

## Description

Call Me Maybe is an introduction to function calling with large language
models. It converts natural-language requests into machine-readable function
calls by selecting one function from a supplied schema and extracting all of
its arguments with the required primitive types.

The project uses the Qwen3-0.6B model through the provided `llm_sdk`. A finite
state machine controls generation so the model cannot emit an unknown function
name, malformed JSON structure, missing argument, or value that violates the
selected parameter's grammar. The result is one JSON object per input prompt,
ready to be consumed by ordinary Python code.

The implementation supports these parameter types:

- `string`;
- `number`;
- `integer`;
- `boolean`.

## Instructions

### Requirements

- Python 3.10 or newer;
- [`uv`](https://docs.astral.sh/uv/) for dependency management;
- enough memory to load Qwen3-0.6B through `llm_sdk`.

Install the project and its development dependencies:

```bash
make install
```

Run it with the default files from `data/input/`:

```bash
make run
```

The default result is written to
`data/output/function_calling_results.json`. The directory is created when it
does not exist.

Custom input and output paths can be passed on the command line:

```bash
uv run python -m src \
  --functions_definition path/to/functions_definition.json \
  --input path/to/function_calling_tests.json \
  --output path/to/function_calling_results.json
```

Available development commands:

```bash
make lint          # Run flake8 and the configured mypy checks
make lint-strict   # Run flake8 and mypy --strict
make debug         # Start the application under pdb
make clean         # Remove local Python, test, mypy, and uv caches
```

During execution, the program prints model-loading status and the current
prompt number. A successful run prints the destination of the generated JSON
file. Validation and generation failures are written to standard error and
produce a non-zero exit status.

## Input and output formats

The function-definition file contains a JSON array. Every entry provides a
function name, a natural-language description, its parameter schema, and its
return schema:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

The prompt file is another JSON array. Each object contains exactly one
natural-language request:

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

For every prompt, the application writes its original text, the selected
function name, and all required parameters. It does not add prose or extra
keys:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

Both input files are validated with Pydantic before the model is loaded. Empty
input arrays, unsupported types, malformed JSON, inaccessible paths, and an
unwritable output destination are reported as errors.

## Algorithm explanation

The generation pipeline combines model inference with deterministic structure
and token-level constraints.

1. **Validate the inputs.** Pydantic parses the function definitions and user
   prompts. The program checks the output path before loading the model.
2. **Build the model prompt.** Function names, descriptions, and parameter
   schemas are serialized into the system message. The natural-language
   request is added as the user message.
3. **Tokenize function names.** Every available name is encoded with the custom
   Qwen byte-pair tokenizer and inserted into a token trie.
4. **Select a function.** While the model generates the `name` value, only trie
   edges reachable from the current node remain enabled. Terminal nodes allow
   a name to finish, including when one valid name is a prefix of another.
5. **Emit fixed JSON structure.** Keys, quotes, colons, commas, and braces whose
   positions are already known are appended deterministically by the finite
   state machine.
6. **Constrain every argument.** The selected function schema determines the
   token mask for each parameter:
   - strings accept content tokens and a closing quote; characters requiring
     JSON escaping are encoded safely before being written;
   - regex arguments may close only after the accumulated expression can be
     compiled;
   - numbers use a JSON-number prefix parser supporting signs, fractions, and
     exponents;
   - integers use the numeric parser with fractions and exponents disabled;
   - booleans can follow only prefixes of `true` and `false`.
7. **Mask model logits.** The model returns a score for every vocabulary token.
   Forbidden scores become negative infinity, so `argmax` can select only a
   token accepted by the current FSM state.
8. **Finish complete values.** A small terminator bias helps the model close a
   complete string or fractional number. Per-value and per-prompt token limits
   prevent unbounded generation.
9. **Normalize the result.** Schema `number` values are serialized as floating
   point values. When the model slightly changes a string that is explicitly
   present in the request, reliable source matches restore quoted text,
   parameter text after a colon, and original backslashes.
10. **Parse and write JSON.** The generated token sequence is parsed with
    Python's `json` module before it is added to the output array. Invalid JSON
    therefore cannot be silently written as a successful result.

## Design decisions

### Trie-based function selection

A token trie enforces the complete set of valid names without comparing a
partially generated string against every function on every step. It also works
with the model's actual token boundaries instead of assuming one character per
token.

### FSM-owned JSON structure

The model decides semantic content, while the FSM writes predictable syntax.
This reduces the number of inference steps and removes common formatting
failures such as missing braces, unexpected keys, and trailing commas.

### Grammar-based primitive values

Numeric constraints validate both partial and complete forms. For example,
`-`, `1.`, and `2e-` are legal prefixes but cannot terminate; `-4`, `1.25`, and
`2e-3` are complete values. Boolean generation uses the same prefix principle
with the finite targets `true` and `false`.

### Semantic strings and serialized strings

String content and its JSON representation have different escaping rules. A
semantic regex such as `\d+` must be stored as `"\\d+"` in JSON. Escaping only
the token being appended preserves the semantic value while keeping the output
parseable. Source-text recovery is deliberately limited to strong matches so
derived values, such as regular expressions, remain model-generated.

### Token cache and actual delimiter IDs

The vocabulary is classified once per tokenizer instance and reused across all
prompts. Delimiter token IDs are resolved through the tokenizer instead of
being hardcoded, which keeps the FSM consistent with the supplied vocabulary.

## Project structure

```text
.
├── data/input/           Example function definitions and prompts
├── llm_sdk/              Model interface supplied with the subject
├── src/
│   ├── fsm.py            State machine and constrained token masks
│   ├── json_generator.py Result parsing and parameter normalization
│   ├── main.py           CLI orchestration and error handling
│   ├── prompt_generator.py
│   ├── tokeniser.py      Custom Qwen byte-pair tokenizer
│   ├── trie.py           Token trie for function names
│   └── validation.py     Pydantic input models and path checks
├── Makefile
├── pyproject.toml
└── uv.lock
```

Generated output and local caches are excluded from version control.

## Performance analysis

Model inference is the dominant cost because every model-selected token needs
a new logits request. Trie traversal is linear in the number of tokens in the
chosen function name. Numeric validation scans only the preclassified numeric
subset of the vocabulary, and the cached type-specific token sets avoid a full
vocabulary scan for every prompt.

The earlier developer-reported private moulinette score was 11/11. That suite
is not included here and was not rerun for this maintenance revision.
It does not establish general semantic accuracy.

The public offline suite is reproducible without weights, the SDK or API calls:

```sh
python -m pip install numpy pydantic regex
python -m unittest discover -s tests -v
```

With the full development environment, use `make test` and `make lint-strict`.
The tests use scripted logits and a tiny tokenizer to check JSON numeric
prefixes, integer/float/Boolean decoding, escaped names/keys/values, trie prefix
names, empty parameters, limits, masking and invalid input schemas.
They do not test real-model function selection or accuracy.

The supported launch command is `uv run python -m src`; this workspace is not
installed as a console-script package. The stale console entry point was removed.
No real-model inference or model download was performed for this revision.

## Challenges faced

### JSON and regular-expression escaping

A backslash has a meaning in both regex syntax and JSON syntax. Writing the
semantic regex directly produced invalid JSON escape sequences such as `\d`.
The solution keeps semantic tokens separate from serialized tokens and applies
JSON escaping before output.

### Numeric termination

After generating a valid decimal such as `0.0375`, the model could continue
adding zeros indefinitely because every longer value remained grammatically
valid. Completion bias now favors the real comma or closing-brace token once a
fraction is complete, and hard token limits provide a second safeguard.

### Byte-level tokenization

Qwen's vocabulary represents bytes through printable Unicode code points, and
a token does not always correspond to one character. The tokenizer therefore
reimplements byte-to-Unicode conversion and ranked byte-pair merges. Tokens are
decoded and re-encoded only when JSON escaping requires it.

### Exact source strings

Small models may drop punctuation, replace Windows backslashes with forward
slashes, or add a short suffix to copied text. The normalization stage uses
only reliable evidence from the original request to restore exact source text
without hardcoding test values.

## Testing strategy

The public offline suite checks validation, grammar, trie prefix names, escaping,
masking, empty parameters and token limits with scripted logits. Historical SDK
tokenizer diagnostics and private end-to-end scores are developer reports and
have not been reproduced for this maintenance revision. Passing offline tests
does not establish real-model semantic accuracy.

Run the local static checks with:

```bash
make lint-strict
uv run python -m compileall -q src
```

Run an end-to-end generation against any compatible dataset with:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output /tmp/function_calling_results.json
```

## Example usage

Given this request:

```json
{"prompt": "Replace all numbers in \"Room 42\" with NUM"}
```

and a matching `fn_substitute_string_with_regex` definition, the generated
entry has the required structure and properly escaped regex argument:

```json
{
  "prompt": "Replace all numbers in \"Room 42\" with NUM",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Room 42",
    "regex": "\\d+",
    "replacement": "NUM"
  }
}
```

In Python, parsing this JSON recovers the semantic regex `\d+`, which can be
passed directly to `re.sub`.

## Resources

- [RFC 8259: The JavaScript Object Notation Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Python `re` documentation](https://docs.python.org/3/library/re.html)
- [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Sennrich, Haddow, and Birch: Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- Hopcroft, Motwani, and Ullman, *Introduction to Automata Theory, Languages,
  and Computation*, for finite-state machines and formal languages.

### Use of AI

AI was used as a development assistant for code review, identifying edge cases
in the FSM, diagnosing numeric non-termination and JSON escaping failures,
suggesting test cases, writing Google-style docstrings, and improving this
README.

Maintenance fixes and the public offline suite were prepared with OpenAI Codex on 2026-10-21.
