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
    