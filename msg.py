from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from typing import List, Dict
import boto3  # Make sure you have boto3 installed

def create_step_back_prompt(initial_list: List[str]) -> str:
    """Creates a prompt to ask the LLM for easily extractable values."""
    prompt = f"""You are a helpful AI assistant. 
    From the following list of data to extract: {', '.join(initial_list)}, 
    tell me which values can be easily extracted without needing complex processing. 
    Provide your answer as a comma-separated list."""
    return prompt

def extract_and_remove_easy_values(llm_response: str, initial_list: List[str]) -> Dict:
    """Extracts values from the LLM response and removes them from the initial list."""
    extracted_values = [value.strip() for value in llm_response.split(",") if value.strip()]
    remaining_list = [value for value in initial_list if value not in extracted_values]
    return {value: "easily extracted" for value in extracted_values}, remaining_list

def create_fewshot_prompt(remaining_list: List[str], examples: List[Dict]) -> PromptTemplate:
    """Creates a few-shot prompt for the remaining values."""
    prompt_text = f"""You are a helpful AI assistant. 
    Extract the following values from the provided text: {', '.join(remaining_list)}

    Here are some examples:
    """
    for example in examples:
        prompt_text += f"Text: {example['text']}\n"
        prompt_text += f"Values: {', '.join([f'{key}: {value}' for key, value in example['values'].items()])}\n\n"
    prompt_text += "Text: {text}\nValues:"
    return PromptTemplate(
        input_variables=["text"],
        template=prompt_text
    )

def main():
    # 1. Define your initial list of values and few-shot examples
    initial_list = ["trade_date", "execution_date", "amount", "customer_name", "complex_calculation_result"]
    fewshot_examples = [
        {"text": "The trade date is 2023-10-26 and the amount is $100.", "values": {"trade_date": "2023-10-26", "amount": "$100"}},
        # ... more examples ...
    ]

    # 2. Initialize the Bedrock LLM
    # Make sure to replace 'your_bedrock_region' with your actual region
    llm = Bedrock(model_id="anthropic.claude-v2", client=boto3.client('bedrock-runtime', region_name='your_bedrock_region')) 

    # --- Create the Sequential Chain ---

    # 1. Step-back chain to identify easy values
    step_back_prompt = create_step_back_prompt(initial_list)
    step_back_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=step_back_prompt, input_variables=[]))

    # 2. Few-shot chain for remaining values (depends on the output of step_back_chain)
    def fewshot_chain_factory(easy_values_response: Dict):
        # Extract the LLM output from the dictionary
        easy_values_response = easy_values_response['text']  
        extracted_easy_values, remaining_list = extract_and_remove_easy_values(easy_values_response, initial_list)
        fewshot_prompt = create_fewshot_prompt(remaining_list, fewshot_examples)
        return LLMChain(llm=llm, prompt=fewshot_prompt)

    # 3. Combine the chains using SequentialChain
    overall_chain = SequentialChain(
        chains=[
            step_back_chain,
            fewshot_chain_factory,  # Pass the factory function
        ],
        input_variables=["text"],  # Input to the first chain (step_back_chain)
        output_variables=["extracted_values"]  # Output from the last chain (fewshot_chain)
    )

    # --- Run the Sequential Chain ---
    # Provide the input text to the overall_chain
    input_text = "The trade date is 2024-01-20. The customer name is Alice. Some complex calculation result is 42." 
    result = overall_chain({"text": input_text})

    # --- Process and print the results ---
    extracted_values = {}
    # Extract values from the first chain (easy values)
    easy_values_output = result[0]['text']  # Extract the text output from the dictionary
    extracted_easy_values, _ = extract_and_remove_easy_values(easy_values_output, initial_list)
    extracted_values.update(extracted_easy_values)

    # Extract values from the second chain (few-shot)
    fewshot_values_output = result[1]['text']
    # ... (Implement your logic to parse fewshot_values_output and extract values into a dictionary) ...
    extracted_values.update(extracted_fewshot_values)  

    print(extracted_values) 

if __name__ == "__main__":
    main()
