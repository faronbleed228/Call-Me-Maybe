import llm_sdk


def main():
    LLM_Model = llm_sdk.Small_LLM_Model()
    print(LLM_Model.get_path_to_tokenizer_file())


if __name__ == "__main__":
    main()
