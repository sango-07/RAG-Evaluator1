import torch
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams
import pandas as pd

import torch
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams
import pandas as pd

def RAGEvaluator(df, selected_metrics):
    # Load models and pipelines
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    bias_pipeline = pipeline("zero-shot-classification", model="Hate-speech-CNERG/dehatebert-mono-english")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Define metric evaluation functions
    def evaluate_bleu_rouge(candidates, references):
        bleu_score = corpus_bleu(candidates, [references]).score
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return bleu_score, rouge1

    def evaluate_bert_score(candidates, references):
        P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
        return P.mean().item(), R.mean().item(), F1.mean().item()

    def evaluate_perplexity(text):
        encodings = gpt2_tokenizer(text, return_tensors='pt')
        max_length = gpt2_model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    def evaluate_diversity(texts):
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score

    def evaluate_racial_bias(text):
        results = bias_pipeline([text], candidate_labels=["hate speech", "not hate speech"])
        bias_score = results[0]['scores'][results[0]['labels'].index('hate speech')]
        return bias_score

    # Process each row and add selected metric results to the DataFrame
    for idx, row in df.iterrows():
        question, answer, contexts = row['question'], row['answer'], row['contexts']
        candidates = [answer]
        references = [contexts]

        # Calculate metrics as per the selected metrics list and add them as columns in the DataFrame
        if "BLEU" in selected_metrics or "ROUGE-1" in selected_metrics:
            bleu, rouge1 = evaluate_bleu_rouge(candidates, references)
            if "BLEU" in selected_metrics:
                df.at[idx, "BLEU"] = bleu
            if "ROUGE-1" in selected_metrics:
                df.at[idx, "ROUGE-1"] = rouge1
        if "BERT Precision" in selected_metrics or "BERT Recall" in selected_metrics or "BERT F1" in selected_metrics:
            bert_p, bert_r, bert_f1 = evaluate_bert_score(candidates, references)
            if "BERT Precision" in selected_metrics:
                df.at[idx, "BERT Precision"] = bert_p
            if "BERT Recall" in selected_metrics:
                df.at[idx, "BERT Recall"] = bert_r
            if "BERT F1" in selected_metrics:
                df.at[idx, "BERT F1"] = bert_f1
        if "Perplexity" in selected_metrics:
            df.at[idx, "Perplexity"] = evaluate_perplexity(answer)
        if "Diversity" in selected_metrics:
            df.at[idx, "Diversity"] = evaluate_diversity(candidates)
        if "Racial Bias" in selected_metrics:
            df.at[idx, "Racial Bias"] = evaluate_racial_bias(answer)

    return df


# def RAGEvaluator(df, selected_metrics):
#     # Load models and pipelines
#     gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
#     gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     bias_pipeline = pipeline("zero-shot-classification", model="Hate-speech-CNERG/dehatebert-mono-english")
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
#     # Function definitions for evaluations
#     def evaluate_bleu_rouge(candidates, references):
#         bleu_score = corpus_bleu(candidates, [references]).score
#         rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
#         rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
#         return bleu_score, rouge1

#     def evaluate_bert_score(candidates, references):
#         P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
#         return P.mean().item(), R.mean().item(), F1.mean().item()

#     def evaluate_perplexity(text):
#         encodings = gpt2_tokenizer(text, return_tensors='pt')
#         max_length = gpt2_model.config.n_positions
#         stride = 512
#         lls = []
#         for i in range(0, encodings.input_ids.size(1), stride):
#             begin_loc = max(i + stride - max_length, 0)
#             end_loc = min(i + stride, encodings.input_ids.size(1))
#             trg_len = end_loc - i
#             input_ids = encodings.input_ids[:, begin_loc:end_loc]
#             target_ids = input_ids.clone()
#             target_ids[:, :-trg_len] = -100
#             with torch.no_grad():
#                 outputs = gpt2_model(input_ids, labels=target_ids)
#                 log_likelihood = outputs[0] * trg_len
#             lls.append(log_likelihood)
#         ppl = torch.exp(torch.stack(lls).sum() / end_loc)
#         return ppl.item()

#     def evaluate_diversity(texts):
#         all_tokens = [tok for text in texts for tok in text.split()]
#         unique_bigrams = set(ngrams(all_tokens, 2))
#         diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
#         return diversity_score

#     def evaluate_racial_bias(text):
#         results = bias_pipeline([text], candidate_labels=["hate speech", "not hate speech"])
#         bias_score = results[0]['scores'][results[0]['labels'].index('hate speech')]
#         return bias_score

#     # Dictionary to store results for each metric per row
#     metrics_data = {metric: [] for metric in selected_metrics}

#     # Evaluate each row in the DataFrame
#     for idx, row in df.iterrows():
#         question, answer, contexts = row['question'], row['answer'], row['contexts']
#         candidates = [answer]
#         references = [contexts]

#         # Collect metrics conditionally based on selected_metrics
#         if 'BLEU' in selected_metrics or 'ROUGE-1' in selected_metrics:
#             bleu, rouge1 = evaluate_bleu_rouge(candidates, references)
#             if 'BLEU' in selected_metrics:
#                 metrics_data['BLEU'].append(bleu)
#             if 'ROUGE-1' in selected_metrics:
#                 metrics_data['ROUGE-1'].append(rouge1)

#         if 'BERT Precision' in selected_metrics or 'BERT Recall' in selected_metrics or 'BERT F1' in selected_metrics:
#             bert_p, bert_r, bert_f1 = evaluate_bert_score(candidates, references)
#             if 'BERT Precision' in selected_metrics:
#                 metrics_data['BERT Precision'].append(bert_p)
#             if 'BERT Recall' in selected_metrics:
#                 metrics_data['BERT Recall'].append(bert_r)
#             if 'BERT F1' in selected_metrics:
#                 metrics_data['BERT F1'].append(bert_f1)

#         if 'Perplexity' in selected_metrics:
#             perplexity = evaluate_perplexity(answer)
#             metrics_data['Perplexity'].append(perplexity)

#         if 'Diversity' in selected_metrics:
#             diversity = evaluate_diversity(candidates)
#             metrics_data['Diversity'].append(diversity)

#         if 'Racial Bias' in selected_metrics:
#             racial_bias = evaluate_racial_bias(answer)
#             metrics_data['Racial Bias'].append(racial_bias)

#     # Convert metrics_data dictionary to a DataFrame
#     metrics_df = pd.DataFrame(metrics_data)

#     # Concatenate original DataFrame with metrics DataFrame
#     result_df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

#     return result_df