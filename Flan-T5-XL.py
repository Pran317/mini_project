"""
This program is build with Flan-T5-XL LLM to be able to determine whether an answer to a query is correct or not. 

It accepts two parameters provided as a command line input. 
The first input represents the query and the second input is the answer. 
If the answer is correct for the query then the output should be 'YES' else 'NO'.

Syntax: python template.py <string> <string>

The following example is given for your reference:

Terminal: python template.py 'Who is Rabindranath Tagore?' 'He is a very famous football player.'
  Output: NO

Terminal: python template.py 'Who is Rabindranath Tagore?' 'He is a famous poet, writer, playwright, composer, philosopher, social reformer, and painter of the Bengal Renaissance.'
  Output: YES

You are expected to create some examples of your own to test the correctness of your pipeline.
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * 
"""
def clean_decoded_output(decoded_output):
    """
    Cleans the output from the LLM to return either 'YES' or 'NO'.
    It handles variations in LLM output to ensure a consistent binary response.
    """
    cleaned = decoded_output.strip().upper()
    if "YES" in cleaned:
        return "YES"
    elif "NO" in cleaned:
        return "NO"
    else:
        # Fallback if neither 'YES' nor 'NO' is clearly present in the output.
        # This ensures a 'NO' if the model gives an unexpected response.
        return "NO"

def llm_function(model, tokenizer, query, answer):
    '''
    This function formulates a prompt, tokenizes it, sends it to the Flan-T5-XL model,
    decodes the output, and then cleans it to return 'YES' or 'NO'.

    1. Properly formulate the prompt as per the question - which should output either 'YES' or 'NO'.
       The output must always be upper-case. You may post-process to get the desired output.
    2. Tokenize the prompt.
    3. Pass the tokenized prompt to the model and generate output. You can use default hyper-parameters for the model.
    4. Decode the model output.
    5. Clean output and return.
    '''
    # 1. Properly formulate the prompt for correctness evaluation.
    # The prompt explicitly asks the model to evaluate the answer based on the query,
    # expecting a YES/NO response.
    prompt = f"Query: {query}\nAnswer: {answer}\nIs the answer correct for the query? Please respond with YES or NO."

    # 2. Tokenize the prompt.
    # 'return_tensors="pt"' ensures the output is PyTorch tensors.
    inputs = tokenizer(prompt, return_tensors="pt")

    # 3. Pass the tokenized prompt to the model and generate output.
    # 'max_new_tokens' limits the length of the generated response,
    # 'temperature' controls the randomness (lower means more deterministic),
    # 'do_sample=True' enables sampling for generation.
    output_tokens = model.generate(**inputs, max_new_tokens=5, temperature=0.1, do_sample=True)

    # 4. Decode the model output.
    # 'skip_special_tokens=True' removes any special tokens (like [CLS], [SEP])
    # that are part of the tokenization process but not actual text.
    decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # 5. Clean output and return.
    cleaned_output = clean_decoded_output(decoded_output)

    return cleaned_output



"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':
    input_data_one = sys.argv[1].strip()
    input_data_two = sys.argv[2].strip()

    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    
    a = input_data_one
    b = input_data_two
    
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,a,b)
    print(out.strip())

    """ End to call """