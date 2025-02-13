# Creates the LLM to be trained, along with querying its dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch

# Sampling parameters class for LanguageModel
class SamplingParameters:
    def __init__(self, max_new_tokens=50, max_length=512, min_length=0, temperature=0.9, top_k=50, top_p=1.0, num_return_sequences=1, do_sample=True, num_beams=1, best_of=1, pad_token_id=None, eos_token_id=None):
        # The maximum length of the tokenized sequence.    
        self.max_length = max_length
        # The maximum length of the generated sequence.
        self.max_new_tokens = max_new_tokens
        # The minimum length of the generated sequence.
        self.min_length = min_length
        # Controls the randomness of predictions by scaling the logits before applying softmax.
        self.temperature = temperature
        # Limits the sampling pool to the top-k most probable tokens.
        self.top_k = top_k
        # Filters the sampling pool to include only tokens whose cumulative probability is <= top_p.
        self.top_p = top_p
        # The number of output sequences to generate.
        self.num_return_sequences = num_return_sequences
        # Penalizes repeated tokens to reduce repetitiveness in the output.
        self.repetition_penalty = 1.0
        # If True, sampling is used for text generation; if False, greedy decoding is used.
        self.do_sample = do_sample
        # Stops generation once an EOS token is generated for all beams.
        self.early_stopping = True
        # The number of beams for beam search; higher values improve quality but increase computation.
        self.num_beams = num_beams
        # The number of best sequences returned when using beam search.
        self.best_of = best_of
        # The token ID used for padding shorter sequences.
        self.pad_token_id = pad_token_id
        # The token ID representing the end of a sequence.
        self.eos_token_id = eos_token_id
        


def generateWithTokens(self, tokens, return_probabilites=True, **kwargs):            
    # Generate responses using the model with the provided sampling parameters.
    output = self.model.generate(**tokens,**kwargs,output_scores=True,return_dict_in_generate=True)
        
    # Get transition / per-token scores
    transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
    input_size = tokens.shape[1]
    # Get tokens without input prompt
    gen_tokens = output.sequences[:,input_size:]
    logits = []
    data = []
    # Match per-token logits / scores with each respective output token 
    for i in range(len(gen_tokens)):
        data = zip(gen_tokens[i], transition_scores[i])
        logits.append([])
        # Add to logits, convert to probabilities if specified
        for token, score in data:
            logits[i].append(np.exp(score.cpu().numpy()).item() if return_probabilites else score.cpu().numpy().item())
        data.append({"output": gen_tokens[i], "confidence": logits[i] })
    # Return formatted output tokens
    return data
    

    # Creates a tree consisting of equal chain of thought lengths and nodes, with the output represented as tokens
    def CoTTreeTokens(self, samplingParams : SamplingParameters, prompt, length, array=None, chain_text="", attention_mask=None):
        # End case of recursive function
        if length < 1:
            return

        # Accounts for initial prompt
        if array == None:
            array = [prompt]
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=samplingParams.max_length).to(self.device)
            chain_text = tokens.input_ids
            attention_mask = tokens.attention_mask
            next_text = chain_text
        else:

            next_text = torch.cat((chain_text, prompt.unsqueeze(0)), dim=1)    

        # Generate each node that contains a sequence, defined in samplingParams by the num_return_sequences
        output = self.generateWithTokens(next_text,samplingParams,True,attention_mask)
        
        # Loop through each node
        for sequence in output:
            # Add sequence to return data
            array.append([sequence])
            # Recursive method, constructs the CoT chain data coming off each node
            # TODO: could add in modifications to the number of nodes as depth of tree increases based on set number or confidence levels...
            self.CoTTreeTokens(samplingParams, sequence["output"], length - 1, array[len(array)-1], next_text)
        # Return resulting array, only called once after all recursive functions have returned
        return array

    def generate(self, prompt, samplingParams: SamplingParameters, return_probabilites=False, return_tokens=False, custom_attention_mask=None):
        # Tokenize the input prompt, ensuring proper padding and truncation.
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=samplingParams.max_length).to(self.device)
        
        # Use a custom attention mask if provided; otherwise, use the default attention mask.
        attention_mask = custom_attention_mask if custom_attention_mask is not None else inputs.attention_mask

        # Generate responses using the model with the provided sampling parameters.
        output = self.model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=samplingParams.max_new_tokens,
            min_length=samplingParams.min_length,
            temperature=samplingParams.temperature,
            top_k=samplingParams.top_k,
            top_p=samplingParams.top_p,
            num_return_sequences=samplingParams.num_return_sequences,
            repetition_penalty=samplingParams.repetition_penalty,
            do_sample=samplingParams.do_sample,
            early_stopping=samplingParams.early_stopping if samplingParams.num_beams > 1 else False,
            num_beams=samplingParams.num_beams,
            pad_token_id=samplingParams.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=samplingParams.eos_token_id or self.tokenizer.eos_token_id,
            # Forces scores / logits to be output
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Get transition / per-token scores
        transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        input_size = inputs.input_ids.shape[1]
        # Get tokens without input prompt
        gen_tokens = output.sequences[:,input_size:]
        logits = []
        # Match per-token logits / scores with each respective output token 
        for i in range(samplingParams.num_return_sequences):
            data = zip(gen_tokens[i], transition_scores[i])
            logits.append([])
            # Add to logits, convert to probabilities if specified
            for token, score in data:
                logits[i].append(np.exp(score.cpu().numpy()).item() if return_probabilites else score.cpu().numpy().item())
        # Return formatted output tokens (faster) or decoded output 
        if return_tokens:
            return [{"output": gen_tokens[i], "confidence": logits[i] } for i in range(samplingParams.num_return_sequences)]    
        return [{"output": self.tokenizer.decode(gen_tokens[i],skip_special_tokens=True), ("confidence"): logits[i] } for i in range(samplingParams.num_return_sequences)]
    
    # Creates a tree consisting of equal chain of thought lengths and nodes, with the output represented as a string (tokenized is faster, TODO)
    def CoTTreeStrings(self, samplingParams : SamplingParameters, prompt, length, array=None, chain_text=""):
        # End case of recursive function
        if length < 1:
            return
        
        # Accounts for initial prompt
        if array == None:
            array = [prompt]
            
        # Extends prompt 
        next_text = chain_text+prompt
        # Generate each node that contains a sequence, defined in samplingParams by the num_return_sequences
        output = self.generate(next_text,samplingParams,True)
        
        # Loop through each node
        for sequence in output:
            # Add sequence to return data
            array.append([sequence])
            # Recursive method, constructs the CoT chain data coming off each node
            # TODO: could add in modifications to the number of nodes as depth of tree increases based on set number or confidence levels...
            self.CoTTreeStrings(samplingParams, sequence["output"], length - 1, array[len(array)-1], next_text)
        # Return resulting array, only called once after all recursive functions have returned
        return array

