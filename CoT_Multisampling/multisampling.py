import numpy as np
import torch

def mutlisampleGenerate(model, tokens, return_probabilites=True, **kwargs):
    """
    Generate multiple samples from a language model and compute token-level confidence scores.

    Args:
        model: The language model to use for generation (must support generate() method)
        tokens (torch.Tensor): Input token ids of shape [batch_size, sequence_length]
        return_probabilites (bool): If True, convert logits to probabilities. If False, return raw logits
        **kwargs: Additional keyword arguments passed to model.generate()
            Common arguments include:
            - temperature (float): Sampling temperature (higher = more random)
            - top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            - num_return_sequences (int): Number of sequences to generate
            - do_sample (bool): Whether to use sampling vs greedy decoding
            - max_new_tokens (int): Maximum number of new tokens to generate
            - num_beams (int): Number of beams for beam search
            - pad_token_id: Token ID used for padding
            - eos_token_id: Token ID that signals end of sequence

    Returns:
        list[dict]: List of dictionaries containing:
            - output (torch.Tensor): Generated token ids
            - confidence (list[float]): Confidence scores for each generated token
    """
    # Generate responses using the model with the provided sampling parameters.
    inputs = {'input_ids': tokens}
    output = model.generate(**inputs,**kwargs,do_sample=True,output_scores=True,return_dict_in_generate=True)
        
    # Get transition / per-token scores
    transition_scores = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
    input_size = tokens.shape[1]
    # Get tokens without input prompt
    gen_tokens = output.sequences[:,input_size:]
    logits = []
    return_data = []
    # Match per-token logits / scores with each respective output token 
    for i in range(len(gen_tokens)):
        data = zip(gen_tokens[i], transition_scores[i])
        logits.append([])
        # Add to logits, convert to probabilities if specified
        for token, score in data:
            logits[i].append(np.exp(score.cpu().numpy()).item() if return_probabilites else score.cpu().numpy().item())
        return_data.append({"output": gen_tokens[i], "confidence": logits[i] })
    # Return formatted output tokens
    return return_data
    

# Creates a tree consisting of equal chain of thought lengths and nodes, with the output represented as tokens
def CoTTreeTokens(model, prompt, length, return_probabilities, **kwargs):
    """
    Generate a tree of chain-of-thought sequences using recursive token generation.

    Args:
        model: The language model to use for generation (must support generate() method)
        prompt (torch.Tensor or list): Input prompt tokens to start generation from
        length (int): Number of steps/levels in each chain of thought
        return_probabilities (bool): Whether to return token probabilities vs raw logits
        **kwargs: Additional keyword arguments passed to mutlisampleGenerate()
            Important arguments include:
            - num_return_sequences (int): Number of branches at each node
            - temperature (float): Sampling temperature
            - top_p (float): Nucleus sampling parameter
            - max_new_tokens (int): Max tokens to generate per step

    Returns:
        list: Nested list structure representing the tree, where each node contains:
            - output (torch.Tensor): Generated token ids for that step
            - confidence (list[float]): Confidence scores for generated tokens

    The function builds a tree where each node branches into num_return_sequences
    children, and each branch extends for 'length' steps, creating chains of
    sequential thoughts.
    """
    def generateNextTokenSequence(model, tokens, length, arr, return_probabilities, **kwargs):
        # End case of recursive function
        if length < 1:
            return

        # Generate each node that contains a sequence, defined in samplingParams by the num_return_sequences
        output = mutlisampleGenerate(model, tokens, return_probabilities, **kwargs)
        
        # Loop through each node
        for sequence in output:
            # Add sequence to return data
            arr.append([sequence])
            # Recursive method, constructs the CoT chain data coming off each node
            # Concatenate the new tokens with the input tokens
            next_tokens = torch.cat((tokens, sequence["output"].unsqueeze(0)), dim=1)
            generateNextTokenSequence(model, next_tokens, length - 1, arr[len(arr)-1], return_probabilities, **kwargs)
    
    # Ensure prompt is properly formatted as a tensor if it isn't already
    if not isinstance(prompt, torch.Tensor):
        prompt = torch.tensor(prompt).unsqueeze(0)
    array = []   
    generateNextTokenSequence(model, prompt, length, array, return_probabilities, **kwargs)
    # Return resulting array, only called once after all recursive functions have returned
    return array