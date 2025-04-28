# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import os
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Callable, Iterable

from kebab.tasks.text_completion import TextCompletionTask


class BaseRAGTextCompleter(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
    ) -> tuple[float, list[list[dict[str, Any]]]]:
        raise NotImplementedError

    def complete_partial_queries(
        self,
        partial_queries_with_augmented_contexts: Iterable[dict[str, Any]],
    ) -> Iterable[dict[str, Any]]:
        count = 0
        for partial_query_with_augmented_context in partial_queries_with_augmented_contexts:
            if partial_query_with_augmented_context["text_with_mask"] == "":
                result = copy.deepcopy(partial_query_with_augmented_context)
                yield result
                continue

            target_content_logprob, target_content_top_logprobs = self.complete_single_partial_query(
                partial_query_with_augmented_context["text_with_mask"],
                partial_query_with_augmented_context["target_content"],
                partial_query_with_augmented_context["augmented_context"],
            )

            result = copy.deepcopy(partial_query_with_augmented_context)
            result["target_content_logprob"] = target_content_logprob
            result["target_content_top_logprobs"] = target_content_top_logprobs

            count += 1
            print(f"Processed {count} partial queries.")

            yield result

    @staticmethod
    def augment_partial_queries_with_contexts(
        queries: Iterable[dict[str, str]],
        get_augmented_context: Callable[[dict[str, str]], str] = lambda _: "",
    ) -> Iterable[dict[str, str]]:
        for query in queries:
            query["augmented_context"] = get_augmented_context(query)
            yield query

    @staticmethod
    def get_font_color(logprob):
        """
        Compute the font color based on log probability.
        """
        if logprob > -4:
            return "#000000"  # Black for high confidence
        elif logprob > -8:
            return "#800000"  # Dark red for medium confidence
        else:
            return "#FF0000"  # Bright red for low confidence

    @staticmethod
    def generate_annotated_doc_html(text_completion_results, output_path):
        """
        Create an HTML page that highlights words using font color based on log probability.
        """
        html_content = """
<html>
    <body>
        <p>LLM Prediction Confidence Visualization: Words are color-coded based on their log probabilities:</p>
        <ul>
            <li><span style="color: black;">High confidence (black):</span> logprob > -4</li>
            <li><span style="color: #800000;">Medium confidence (dark red):</span> -8 < logprob <= -4</li>
            <li><span style="color: red;">Low confidence (red):</span> -12 <= logprob <= -8</li>
            <li><span style="color: red; font-weight: bold;">Very low confidence (bold red):</span> logprob <= -12</li>
        </ul>
        <hr>
        <p>
        """
        for item in text_completion_results:
            if item["text_with_mask"] == "":
                html_content += f'<span class="word">{item["target_content"]}</span>'
                continue
            target_word_logprob = item["target_content_logprob"]
            tooltip = ""
            if target_word_logprob == float("-inf"):
                tooltip = "Incorrect prediction, "
                target_word_logprob = item["target_content_top_logprobs"][0][-1]["logprob"]
            color = BaseRAGTextCompleter.get_font_color(target_word_logprob)
            font_weight = "bold" if target_word_logprob <= -12 else "normal"
            tooltip += f"LogProb: {target_word_logprob:.3f}"
            top_logprobs_html = str(item["target_content_top_logprobs"]).replace('"', '&quot;')
            tooltip += f', top {len(item["target_content_top_logprobs"][0])} token predictions: {top_logprobs_html}'
            html_content += f'<span class="word" style="color: {color}; font-weight: {font_weight};" title="{tooltip}">{item["target_content"]}</span>'
        html_content += """
        </p>
    </body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML file generated: {os.path.abspath(output_path)}")


class PhiRAGTextCompleter(BaseRAGTextCompleter):

    def __init__(self) -> None:
        model_id = "microsoft/phi-4"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bad_token_ids = [
            self.tokenizer.convert_tokens_to_ids(token) for token in ["<|im_start|>", "<|im_end|>"]
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(self.device)
        self.model.eval()

    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
        top_k: int = 5,
    ) -> tuple[float, list[list[dict[str, Any]]]]:
        text_completion_prompt = f"""
You are a professional language model trained to assist with text completion tasks. Your goal is to accurately fill in missing parts of text using your understanding of language and context.

Below is the context retrieved to help you make a better prediction. It is marked between <Begin Context> and <End Context>:
<Begin Context>
{augmented_context}
<End Context>

Now, complete the following text by predicting the missing word, represented by "{TextCompletionTask.MASK}". The text is marked between <Begin Text> and <End Text>:
<Begin Text>
{text_with_mask} <the rest of the text is not visible to you>
<End Text>
What is the **single alphanumeric word** that best completes the masked position "{TextCompletionTask.MASK}"? Please respond strictly with **only** the word after "Answer: ".

Answer: """

        # Encode the prompt
        input_ids = self.tokenizer.encode(text_completion_prompt, return_tensors="pt").to(self.device)

        # Get model outputs (logits for each token)
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Extract logits for the next token (last position)
        next_token_logits = outputs.logits[0, -1, :]

        for token_id in self.bad_token_ids:
            next_token_logits[token_id] = float("-inf")

        # Compute log probabilities over the vocabulary
        log_probs = F.log_softmax(next_token_logits, dim=-1)

        # Extract top k token indices and their log probabilities
        top_logprobs, top_indices = torch.topk(log_probs, top_k)
        top_words = [self.tokenizer.decode([idx]) for idx in top_indices]
        top_words = [word.strip() for word in top_words]

        if target_content in top_words:
            target_index = top_words.index(target_content)
            masked_content_logprob = top_logprobs[target_index].item()
        else:
            masked_content_token_ids = self.tokenizer.encode(' ' + target_content, add_special_tokens=False)
            word = self.tokenizer.decode([masked_content_token_ids[0]]).strip()
            if len(word) > 0 and word in top_words:
                target_index = top_words.index(word)
                masked_content_logprob = top_logprobs[target_index].item()
            else:
                masked_content_logprob = float("-inf")  # or assign a default value or handle as needed

        # Output results
        top_logprobs = [[{"token": word, "logprob": logprob} for word, logprob in zip(top_words, top_logprobs.tolist())]]
        return masked_content_logprob, top_logprobs
