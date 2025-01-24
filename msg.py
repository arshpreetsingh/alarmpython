from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from typing import List, Dict
import boto3

def create_step_back_prompt(initial_list: List[str]) -> PromptTemplate:
    """Creates a prompt to ask the LLM for easily extractable values."""
    prompt = f"""You are a helpful AI assistant. 
    From the following list of data to extract: {', '.join(initial_list)}, 
    tell me which values can be easily extracted without needing complex processing. 
    Provide your answer as a comma-separated list."""
    return PromptTemplate(template=prompt, input_variables=[])

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
    # ... (Initialize initial_list, fewshot_examples, llm) ...

    # --- Create the Simplified Chain ---

    # 1. Step-back chain to identify easy values
    step_back_chain = LLMChain(llm=llm, prompt=create_step_back_prompt(initial_list))

    # 2. Few-shot chain for remaining values
    fewshot_chain = LLMChain(llm=llm, prompt=create_fewshot_prompt(initial_list, fewshot_examples))  # Use initial_list here

    # 3. Combine the chains using SimpleSequentialChain
    overall_chain = SimpleSequentialChain(
        chains=[
            step_back_chain,
            fewshot_chain,
        ],
        input_variables=["text"],
        output_variables=["easy_values_response", "fewshot_values_response"]  # Get outputs from both chains
    )

    # --- Run the Chain ---
    input_text = "The trade date is 2024-01-20. The customer name is Alice. Some complex calculation result is 42." 
    result = overall_chain({"text": input_text})

    # --- Process and print the results ---
    extracted_values = {}

    # Extract easy values
    easy_values_output = result['easy_values_response']
    extracted_easy_values, _ = extract_and_remove_easy_values(easy_values_output, initial_list)
    extracted_values.update(extracted_easy_values)

    # Extract few-shot values 
    fewshot_values_output = result['fewshot_values_response']
    # ... (Implement your logic to parse fewshot_values_output and extract values into a dictionary) ...
    extracted_values.update(extracted_fewshot_values)  

    print(extracted_values)

if __name__ == "__main__":
    main()











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






from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from typing import List, Dict
import boto3

# ... (Your existing functions: create_step_back_prompt, extract_and_remove_easy_values, create_fewshot_prompt) ...

def create_general_extraction_prompt(initial_list: List[str]) -> PromptTemplate:
    """Creates a general prompt for initial extraction."""
    prompt_text = f"""You are a helpful AI assistant. 
    Extract the following values from the provided text: {', '.join(initial_list)}

    Text: {{text}}
    Values:"""  # The LLM should output the values in a format you can parse
    return PromptTemplate(
        input_variables=["text"],
        template=prompt_text
    )

def create_refinement_prompt(initial_values: Dict) -> PromptTemplate:
    """Creates a refinement prompt based on initial extracted values."""
    prompt_text = """You are a helpful AI assistant. 
    Review the following extracted values and refine them if necessary:

    Initial Values:
    """
    for key, value in initial_values.items():
        prompt_text += f"{key}: {value}\n"
    prompt_text += """
    Refined Values:"""  # The LLM should output the refined values in a format you can parse
    return PromptTemplate(
        input_variables=["initial_values"],
        template=prompt_text
    )

def main():
    # ... (Your existing code: initialize initial_list, fewshot_examples, llm) ...

    # --- Create the Refinement Chain ---

    # 1. Initial extraction chain
    general_extraction_prompt = create_general_extraction_prompt(initial_list)
    initial_extraction_chain = LLMChain(llm=llm, prompt=general_extraction_prompt)

    # 2. Refinement chain
    refinement_chain = SequentialChain(
        chains=[
            initial_extraction_chain,
            LLMChain(llm=llm, prompt=create_refinement_prompt)  # Pass the create_refinement_prompt function
        ],
        input_variables=["text"],
        output_variables=["refined_values"]
    )

    # --- Run the Refinement Chain ---
    input_text = "The trade date is 2024-01-20. The customer name is Alice. Some complex calculation result is 42." 
    result = refinement_chain({"text": input_text})

    # --- Process and print the results ---
    # (You'll need to implement logic to parse the 'refined_values' output and extract the values)
    print(result['refined_values'])

if __name__ == "__main__":
    main()







from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from typing import List, Dict
import boto3

# ... (Your existing functions: create_step_back_prompt, extract_and_remove_easy_values, create_fewshot_prompt) ...

def create_general_extraction_prompt(initial_list: List[str]) -> PromptTemplate:
    """Creates a general prompt for initial extraction."""
    prompt_text = f"""You are a helpful AI assistant. 
    Extract the following values from the provided text: {', '.join(initial_list)}

    Text: {{text}}
    Values:"""  # The LLM should output the values in a format you can parse
    return PromptTemplate(
        input_variables=["text"],
        template=prompt_text
    )

def create_refinement_prompt(initial_values: Dict) -> PromptTemplate:
    """Creates a refinement prompt based on initial extracted values."""
    prompt_text = """You are a helpful AI assistant. 
    Review the following extracted values and refine them if necessary:

    Initial Values:
    """
    for key, value in initial_values.items():
        prompt_text += f"{key}: {value}\n"
    prompt_text += """
    Refined Values:"""  # The LLM should output the refined values in a format you can parse
    return PromptTemplate(
        input_variables=["initial_values"],
        template=prompt_text
    )

def main():
    # ... (Your existing code: initialize initial_list, fewshot_examples, llm) ...

    # --- Create the Refinement Chain ---

    # 1. Initial extraction chain
    general_extraction_prompt = create_general_extraction_prompt(initial_list)
    initial_extraction_chain = LLMChain(llm=llm, prompt=general_extraction_prompt)

    # 2. Refinement chain
    refinement_chain = SequentialChain(
        chains=[
            initial_extraction_chain,
            LLMChain(llm=llm, prompt=create_refinement_prompt)  # Pass the create_refinement_prompt function
        ],
        input_variables=["text"],
        output_variables=["refined_values"]
    )

    # --- Run the Refinement Chain ---
    input_text = "The trade date is 2024-01-20. The customer name is Alice. Some complex calculation result is 42." 
    result = refinement_chain({"text": input_text})

    # --- Process and print the results ---
    # (You'll need to implement logic to parse the 'refined_values' output and extract the values)
    print(result['refined_values'])

if __name__ == "__main__":
    main()



from langchain.chains import RouterChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# Define your schemas for different document types
trade_confirmation_schema = ResponseSchema(
    name="trade_confirmation",
    description="Schema for extracting information from trade confirmation documents",
    properties=[
        ResponseSchema(name="trade_date", description="Date of the trade"),
        ResponseSchema(name="amount", description="Amount of the trade"),
        # ... other properties
    ]
)

# Create output parsers for each schema
trade_confirmation_parser = StructuredOutputParser.from_response_schema(trade_confirmation_schema)

# Define chains for different document types
trade_confirmation_chain = LLMChain(
    llm=llm, 
    prompt=create_trade_confirmation_prompt(),
    output_parser=trade_confirmation_parser 
)
# ... (Chains for other document types) ...

# Create a router chain
def router_function(input_data):
    # Determine the document type (e.g., using keywords or metadata)
    document_type = determine_document_type(input_data["text"])
    if document_type == "trade_confirmation":
        return "trade_confirmation"
    # ... (Handle other document types) ...

router_chain = RouterChain.from_mapping(
    router_function=router_function,
    chains={
        "trade_confirmation": trade_confirmation_chain,
        # ... other chains ...
    }
)

# Run the router chain
result = router_chain.run({"text": "your_input_text"})
print(result)  # The output will be a structured dictionary based on the schema




from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from typing import List, Dict
import boto3

# ... (Your existing functions) ...

def create_validation_prompt(extracted_values: Dict) -> PromptTemplate:
    """Creates a prompt to validate the extracted values."""
    # ... (Implementation to generate a prompt for validation) ...
    pass

def create_summarization_prompt() -> PromptTemplate:
    """Creates a prompt for summarization."""
    # ... (Implementation to generate a summarization prompt) ...
    pass

def create_qa_prompt(summary: str) -> PromptTemplate:
    """Creates a prompt for question answering based on the summary."""
    # ... (Implementation to generate a QA prompt) ...
    pass

def main():
    # ... (Initialize LLM, initial_list, fewshot_examples) ...

    # --- Create the Multi-Stage Chain ---

    # 1. Initial extraction chain
    initial_extraction_prompt = create_step_back_prompt(initial_list)
    initial_extraction_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=initial_extraction_prompt, input_variables=[]))

    # 2. Few-shot extraction chain (using factory function as before)
    def fewshot_chain_factory(easy_values_response: Dict):
        # ... (Extract easy values and create fewshot_chain) ...

    # 3. Validation chain
    validation_chain = LLMChain(llm=llm, prompt=create_validation_prompt)

    # 4. Summarization chain
    summarization_chain = LLMChain(llm=llm, prompt=create_summarization_prompt())

    # 5. Question answering chain
    qa_chain = LLMChain(llm=llm, prompt=create_qa_prompt)

    # Combine all chains into a multi-stage chain
    overall_chain = SequentialChain(
        chains=[
            initial_extraction_chain,
            fewshot_chain_factory,
            validation_chain,
            summarization_chain,
            qa_chain
        ],
        input_variables=["text"],  # Input to the first chain
        output_variables=["answer"]  # Output from the last chain (QA)
    )

    # --- Run the Multi-Stage Chain ---
    input_text = "The trade date is 2024-01-20. The customer name is Alice. Some complex calculation result is 42." 
    result = overall_chain({"text": input_text})

    # --- Print the final answer ---
    print(result['answer'])

if __name__ == "__main__":
    main()






















