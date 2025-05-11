import numpy as np
from tqdm import tqdm
import evaluate
from transformers import GenerationConfig


def get_rouge_results(dataset, model, tokenizer):
    rouge = evaluate.load('rouge')
    dialogues = dataset['test'][0:10]['dialogue']
    human_baseline_summaries = dataset['test'][0:10]['summary']

    model_summaries = []
    for _, dialogue in enumerate(dialogues):
        prompt = f"""
        Summarize the following conversation.
        {dialogue}
        Summary: """
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda').input_ids
        model_outputs = model.generate(
            input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
        model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        model_summaries.append(model_text_output)

    model_results = rouge.compute(
        predictions=model_summaries,
        references=human_baseline_summaries[0:len(model_summaries)],
        use_aggregator=True,
        use_stemmer=True)

    return model_results
    

def get_toxicity_results(toxicity_model_name, text, dataset, model, tokenizer):
    toxicity_evaluator = evaluate.load(
        "toxicity", 
        toxicity_model_name,
        module_type="measurement",
        toxic_label="hate")
    return toxicity_evaluator.compute(predictions=[text])


def evaluate_toxicity(model, tokenizer, toxicity_model_name, dataset, num_samples):
    
    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model (trl model): Model to be evaluated.
    - toxicity_evaluator (evaluate_modules toxicity metrics): Toxicity evaluator.
    - tokenizer (transformers tokenizer): Tokenizer to be used.
    - dataset (dataset): Input dataset for the evaluation.
    - num_samples (int): Maximum number of samples for the evaluation.
        
    Returns:
    tuple: A tuple containing two numpy.float64 values:
    - mean (numpy.float64): Mean of the samples toxicity.
    - std (numpy.float64): Standard deviation of the samples toxicity.
    """

    toxicity_evaluator = evaluate.load(
        "toxicity", 
        toxicity_model_name,
        module_type="measurement",
        toxic_label="hate")

    max_new_tokens=100

    toxicities = []
    input_texts = []
    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]

        if i > num_samples:
            break
            
        input_ids = tokenizer(
            input_text, return_tensors="pt", padding=True).to('cuda').input_ids
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_k=0.0,
            top_p=1.0,
            do_sample=True)
        response_token_ids = model.generate(
            input_ids=input_ids, generation_config=generation_config)
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        toxicity_score = toxicity_evaluator.compute(predictions=[(input_text + " " + generated_text)])
        toxicities.extend(toxicity_score["toxicity"])

    # Compute mean & std using np.
    mean = np.mean(toxicities)
    std = np.std(toxicities)
    return mean, std
