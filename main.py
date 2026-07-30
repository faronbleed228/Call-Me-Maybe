import llm_sdk.llm_sdk


def main():
    LLM_Model = llm_sdk.llm_sdk.Small_LLM_Model()
    print(LLM_Model.get_path_to_merges_file())
    text = "Hi"
    


if __name__ == "__main__":
    main()
