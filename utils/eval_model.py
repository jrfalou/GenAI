import numpy as np
import torch
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


def evaluate_toxicity(model, tokenizer, toxicity_model_name, dataset, num_samples, batch_size=8):
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
        toxic_label="hate"
    )

    max_new_tokens = 100
    input_texts = []
    # Step 1: Accumulate input texts
    for i, sample in enumerate(tqdm(dataset)):
        if i >= num_samples:
            break
        input_texts.append(sample["query"])

    # Step 2: Batch generation
    batches = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]
    completions = []
    for batch in tqdm(batches, desc="Generating responses"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_k=0.0,
            top_p=1.0,
            do_sample=True
        )
        responses = model.generate(**inputs, generation_config=generation_config)
        batch_completions = tokenizer.batch_decode(responses.sequences, skip_special_tokens=True)
        completions.extend(batch_completions)

    # Step 3: Evaluate toxicity in batch
    full_prompts = [q + " " + r for q, r in zip(input_texts, completions)]
    toxicity_scores = toxicity_evaluator.compute(predictions=full_prompts)["toxicity"]

    return np.mean(toxicity_scores), np.std(toxicity_scores)


def evaluate_model(model, eval_dataloader, criterion=None, is_seq2seq=False, metrics=None):
    """
    Unified evaluation function that handles both general and seq2seq model evaluation.
    
    Args:
        model: PyTorch model to evaluate
        eval_dataloader: DataLoader for evaluation data
        criterion: Loss function (optional)
        is_seq2seq: Whether the model is seq2seq (default: False)
        metrics: Dictionary of metric functions to compute (optional)
    
    Returns:
        Dictionary containing evaluation metrics and loss
    """
    model.eval()
    epoch_loss = 0
    metric_values = {name: 0.0 for name in (metrics or {})}
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            label, input_ids = batch['labels'].to('cuda'), batch['input_ids'].to('cuda')
            
            if is_seq2seq:
                output = model(input_ids, label)
                output = output[1:].flatten(0, 1)
                label = label[1:].flatten(0, 1)
            else:
                output = model(input_ids)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(output, label)
                epoch_loss += loss.item()
            
            # Calculate additional metrics if provided
            if metrics:
                for metric_name, metric_fn in metrics.items():
                    metric_values[metric_name] += metric_fn(output, label).item()
    
    # Average the metrics
    results = {
        'loss': epoch_loss / len(eval_dataloader) if criterion is not None else None
    }
    
    if metrics:
        for metric_name in metrics:
            results[metric_name] = metric_values[metric_name] / len(eval_dataloader)
    
    return results