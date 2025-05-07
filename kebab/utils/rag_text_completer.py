# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kebab.tasks.text_completion import TextCompletionTask


class BaseRAGTextCompleter(ABC):
    """
    A base class for RAG text completers, which provides a common interface for different RAG text
    completion models. The derived classes should implement the `complete_single_partial_query`
    method.
    """

    @abstractmethod
    def complete_single_partial_query(
        self,
        text_with_mask: str,
        target_content: str,
        augmented_context: str = "",
    ) -> dict[str, Any]:
        """
        Processes a single partial query and returns the predicted content and log probabilities.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            target_content: The expected content to fill in the mask.
            augmented_context: Additional context to help with the prediction.

        Returns:
            dict[str, Any]: A dictionary requiring the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
        """
        raise NotImplementedError

    def complete_partial_queries(
        self,
        partial_queries: Iterable[dict[str, Any]],
        get_augmented_context: Callable[[dict[str, str]], str],
        verbose: bool = False,
    ) -> Iterable[dict[str, Any]]:
        """
        Processes a collection of partial queries and completes them using the
        `complete_single_partial_query` method.

        Args:
            partial_queries: An iterable of dictionaries, where each dictionary represents a partial
            query to complete.
            get_augmented_context: A function that takes a partial query and returns the augmented
            context as a string.
            verbose: Defaults to False. If True, includes additional information such as the
            original partial query and augmented context in the results for debugging.

        Returns:
            Iterable[dict[str, Any]]: An iterable of dictionaries, where each dictionary contains
            the results for each partial query, including log probabilities and predicted content.
        """
        count = 0
        for query in partial_queries:
            result = {}
            if verbose:
                result = copy.deepcopy(query)

            if query["text_with_mask"] == "":
                # Skip processing if the text with mask is empty; yield the original query dict only
                # if `verbose` is enabled.
                if verbose:
                    yield result
                continue

            # Run RAG to augment a partial query with context.
            augmented_context = get_augmented_context(query)
            if verbose:
                result["augmented_context"] = augmented_context

            # Run `complete_single_partial_query` to complete the partial query.
            result_single_query = self.complete_single_partial_query(
                text_with_mask=query["text_with_mask"],
                target_content=query["target_content"],
                augmented_context=augmented_context,
            )
            result["predicted_content"] = result_single_query["predicted_content"]
            result["target_content_logprob"] = result_single_query["target_content_logprob"]
            if verbose:
                result |= result_single_query

            count += 1
            print(f"Processed {count} partial queries.")

            yield result

    LOGPROB_THRESHOLDS: ClassVar[list[float]] = [-4, -8, -12]

    @staticmethod
    def get_font_color(logprob: float) -> str:
        """Computes the font color based on log probability."""
        if logprob > BaseRAGTextCompleter.LOGPROB_THRESHOLDS[0]:
            return "#000000"  # Black for high confidence
        if logprob > BaseRAGTextCompleter.LOGPROB_THRESHOLDS[1]:
            return "#800000"  # Dark red for medium confidence
        return "#FF0000"  # Bright red for low confidence

    @staticmethod
    def generate_annotated_doc_html(
        text_completion_results: Iterable[dict[str, Any]],
        output_path: str,
    ) -> None:
        """Creates an HTML page that highlights words using font color based on log probability."""
        html_content = f"""
<html>
    <body>
        <p>LLM Prediction Confidence Visualization: Words are color-coded based on their log probabilities:</p>
        <ul>
            <li><span style="color: black;">High confidence (black): </span>logprob > {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[0]}</li>
            <li><span style="color: #800000;">Medium confidence (dark red): </span>{BaseRAGTextCompleter.LOGPROB_THRESHOLDS[1]} < logprob <= {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[0]}</li>
            <li><span style="color: red;">Low confidence (red): </span>{BaseRAGTextCompleter.LOGPROB_THRESHOLDS[2]} < logprob <= {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[1]}</li>
            <li><span style="color: red; font-weight: bold;">Very low confidence (bold red): </span>logprob <= {BaseRAGTextCompleter.LOGPROB_THRESHOLDS[2]}</li>
        </ul>
        <hr>
        <p>
        """
        for result in text_completion_results:
            if result["text_with_mask"] == "":
                html_content += f'<span class="word">{result["target_content"]}</span>'
                continue
            target_word_logprob = result["target_content_logprob"]
            tooltip = ""
            if target_word_logprob == float("-inf"):
                tooltip = "Incorrect prediction, "
                if "predicted_content_top_logprobs" in result:
                    target_word_logprob = result["predicted_content_top_logprobs"][0][-1]["logprob"]
            color = BaseRAGTextCompleter.get_font_color(target_word_logprob)
            font_weight = "bold" if target_word_logprob <= BaseRAGTextCompleter.LOGPROB_THRESHOLDS[2] else "normal"
            tooltip += f"LogProb: {target_word_logprob:.3f}"
            if "predicted_content_top_logprobs" in result:
                top_logprobs_html = str(result["predicted_content_top_logprobs"]).replace('"', "&quot;")
                tooltip += (
                    f", top {len(result['predicted_content_top_logprobs'][0])} token predictions: {top_logprobs_html}"
                )
            html_content += f'<span class="word" style="color: {color}; font-weight: {font_weight};" title="{tooltip}">{result["target_content"]}</span>'
        html_content += """
        </p>
    </body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML file generated: {os.path.abspath(output_path)}")


class PhiRAGTextCompleter(BaseRAGTextCompleter):
    """An implementation of a RAG text completer using the Phi model."""

    TOKENS_TO_EXCLUDE: ClassVar[list[str]] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    """Tokens that are excluded from the model's predictions."""

    def __init__(self, model_id: str = "microsoft/phi-4") -> None:
        """
        Initializes the text completer.

        Args:
            model_id: The Phi model Id to use for text completion.
        """
        super().__init__()
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.token_ids_to_exclude = [
            self.tokenizer.convert_tokens_to_ids(token) for token in PhiRAGTextCompleter.TOKENS_TO_EXCLUDE
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
        top_k: int = 100,
    ) -> dict[str, Any]:
        """
        Processes a single partial query and returns the predicted content and log probabilities.

        Args:
            text_with_mask: The text with a mask indicating the position to be filled.
            target_content: The expected content to fill in the mask.
            augmented_context: Additional context to help with the prediction.
            top_k: The number of top predictions to consider.

        Returns:
            dict[str, Any]: A dictionary containing the following:
                - "predicted_content" (str): The predicted content for the masked position.
                - "target_content_logprob" (float): The log probability of the target content.
                - "predicted_content_top_logprobs" (list[list[dict[str, Any]]]): Each outer list
                contains predicitons at the same token position, and each inner list contains the
                top k predictions for that token position, including the token and its log
                probability.
        """
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

        # Encode the prompt.
        input_ids = self.tokenizer.encode(text_completion_prompt, return_tensors="pt").to(self.device)

        # Get model outputs (logits for each token).
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Extract logits for the next token (last position).
        next_token_logits = outputs.logits[0, -1, :]

        # Exclude bad tokens from the predictions.
        for token_id in self.token_ids_to_exclude:
            next_token_logits[token_id] = float("-inf")

        # Compute log probabilities over the vocabulary.
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        # Extract top k token indices and their log probabilities.
        top_logprobs, top_indices = torch.topk(log_probs, top_k)
        top_tokens = [self.tokenizer.decode([idx]).strip() for idx in top_indices]

        # When the target content contains multiple tokens, LLM can return the target content
        # through multiple paths. For example, LLM can return "Beijing" as two tokens
        # ["Be", "ijing"] or just a single token ["Beijing"]; `self.tokenizer.encode` only returns
        # the former. We will approximately calculate the combined log prob by summing the
        # probabilities of "Beijing" and "Be", where the log prob of "Be" is the upper bound of the
        # log prob of "Be"+"ijing".
        # TODO (allenwang-ms): account for all possible tokenization paths; calculate the accurate
        # log prob of a sequence of tokens by multiple forward passes.
        target_content_id = -1
        target_content_logprob = float("-inf")
        if target_content in top_tokens:
            # First look for the target content as a whole in the top tokens.
            target_content_index = top_tokens.index(target_content)
            target_content_logprob = top_logprobs[target_content_index].item()
            target_content_id = top_indices[target_content_index].item()
        # Second, look for the first token's log prob by tokenizing the target content with the
        # tokenizer.
        target_content_token_ids = self.tokenizer.encode(target_content, add_special_tokens=False)
        if target_content_token_ids[0] != target_content_id:
            first_token_log_prob = log_probs[target_content_token_ids[0]]
            # If the first token is not the same as the target content, we combine the two log probs.
            target_content_logprob = torch.logaddexp(
                torch.tensor(target_content_logprob, device=self.device),
                first_token_log_prob
            ).item()

        # Output results.
        top_logprobs = [
            [
                {"token": word, "logprob": logprob}
                for word, logprob in zip(top_tokens, top_logprobs.tolist(), strict=True)
            ]
        ]

        return {
            "predicted_content": top_logprobs[0][0]["token"],
            "target_content_logprob": target_content_logprob,
            "predicted_content_top_logprobs": top_logprobs,
        }
