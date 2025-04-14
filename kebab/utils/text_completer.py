import copy
import json
import math
import os
import re
import statistics
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from kebab.contracts.document import Document
from kebab.utils.io_helpers import DocumentJsonlReader
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Callable, Iterable


class BaseTextCompleter(ABC):

    mask : str = "<mask>"

    def __init__(self) -> None:
        pass

    @abstractmethod
    def complete_text(
        self,
        text_with_mask: str,
        masked_content: str,
        augmented_context: str = "",
    ) -> tuple[float, list[list[dict[str, Any]]]]:
        raise NotImplementedError

    def complete_text_and_evaluate(
        self,
        partial_queries_with_augmented_contexts: Iterable[dict[str, Any]],
        verbose: bool = False,
    ) -> dict[str, Any]:
        all_logprobs = []
        if verbose:
            all_results = []
        else:
            all_results = None
        count = 0
        for i, partial_query_with_augmented_context in enumerate(partial_queries_with_augmented_contexts):
            if partial_query_with_augmented_context["text_with_mask"] is None:
                if all_results is not None:
                    result = copy.deepcopy(partial_query_with_augmented_context)
                    all_results.append(result)
                    continue

            masked_content_logprob, masked_content_top_logprobs = self.complete_text(
                partial_query_with_augmented_context["text_with_mask"],
                partial_query_with_augmented_context["masked_content"],
                partial_query_with_augmented_context["augmented_context"],
            )
            if masked_content_logprob != float("-inf"):
                all_logprobs.append(masked_content_logprob)
            else:
                all_logprobs.append(masked_content_top_logprobs[0][-1]["logprob"])

            if all_results is not None:
                result = copy.deepcopy(partial_query_with_augmented_context)
                result["masked_content_logprob"] = masked_content_logprob
                result["masked_content_top_logprobs"] = masked_content_top_logprobs
                all_results.append(result)

            count += 1
            print(f"Processed {count} partial queries.")

        if all_logprobs:
            sum_log_prob = sum(all_logprobs)
            avg_log_prob = statistics.mean(all_logprobs)
            var_log_prob = statistics.variance(all_logprobs)
            perplexity = math.exp(-avg_log_prob)
        else:
            raise ValueError("No log probabilities returned.")

        return {
            "sum_log_prob": sum_log_prob,
            "var_log_prob": var_log_prob,
            "count": count,
            "perplexity": perplexity,
            "all_results": all_results,
        }

    @staticmethod
    def generate_partial_queries(docs: Iterable[Document]) -> Iterable[dict[str, str | None]]:
        for doc in docs:
            text = doc.data["text"]
            words = re.findall(r"\w+|\s+|[^\w\s]", text)
            for i, word in enumerate(words):
                if i == 0 or not word.strip().isalnum():
                    yield {
                        "text_with_mask": None,
                        "masked_content": word,
                        "document_id": doc.document_id,
                    }
                    continue
                text_with_mask = "".join(words[ : i] + [BaseTextCompleter.mask])
                yield {
                    "text_with_mask": text_with_mask,
                    "masked_content": word,
                    "document_id": doc.document_id,
                }

    @staticmethod
    def get_font_color(logprob):
        """
        Compute the font color based on log probability.
        - High confidence -> Black (#000000)
        - Medium confidence -> Dark Red (#800000)
        - Low confidence -> Bright Red (#FF0000)
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
            if item["text_with_mask"] is None:
                html_content += f'<span class="word">{item["masked_content"]}</span>'
                continue
            masked_word_logprob = item["masked_content_logprob"]
            tooltip = ""
            if masked_word_logprob == float("-inf"):
                tooltip = "Incorrect prediction, "
                masked_word_logprob = item["masked_content_top_logprobs"][0][-1]["logprob"]
            color = BaseTextCompleter.get_font_color(masked_word_logprob)
            font_weight = "bold" if masked_word_logprob <= -12 else "normal"  # Bold incorrect predictions
            tooltip += f"LogProb: {masked_word_logprob:.3f}"
            top_logprobs_html = str(item["masked_content_top_logprobs"]).replace('"', '&quot;')
            tooltip += f', top {len(item["masked_content_top_logprobs"][0])} token predictions: {top_logprobs_html}'
            html_content += f'<span class="word" style="color: {color}; font-weight: {font_weight};" title="{tooltip}">{item["masked_content"]}</span>'
        html_content += """
        </p>
    </body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML file generated: {os.path.abspath(output_path)}")


class PhiTextCompleter(BaseTextCompleter):

    def __init__(self) -> None:
        model_id = "microsoft/phi-4"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bad_token_ids = [
            self.tokenizer.convert_tokens_to_ids(token) for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(self.device)
        self.model.eval()

    def complete_text(
        self,
        text_with_mask: str,
        masked_content: str,
        augmented_context: str = "",
        top_k: int = 5,
    ) -> tuple[float, list[list[dict[str, Any]]]]:
        text_completion_prompt = f"""
You are a professional language model trained to assist with text completion tasks. Your goal is to accurately fill in missing parts of text using your understanding of language and context.

Below is the context retrieved to help you make a better prediction. It is marked between <Begin Context> and <End Context>:
<Begin Context>
{augmented_context}
<End Context>

Now, complete the following text by predicting the missing word, represented by "{BaseTextCompleter.mask}". The text is marked between <Begin Text> and <End Text>:
<Begin Text>
{text_with_mask} <the rest of the text is not visible to you>
<End Text>
What is the **single alphanumeric word** that best completes the masked position "{BaseTextCompleter.mask}"? Please respond strictly with **only** the word after "Answer: ".

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

        if masked_content in top_words:
            target_index = top_words.index(masked_content)
            masked_content_logprob = top_logprobs[target_index].item()
        else:
            masked_content_token_ids = self.tokenizer.encode(' ' + masked_content, add_special_tokens=False)
            word = self.tokenizer.decode([masked_content_token_ids[0]]).strip()
            if len(word) > 0 and word in top_words:
                target_index = top_words.index(word)
                masked_content_logprob = top_logprobs[target_index].item()
            else:
                masked_content_logprob = float("-inf")  # or assign a default value or handle as needed

        # Output results
        top_logprobs = [[{"token": word, "logprob": logprob} for word, logprob in zip(top_words, top_logprobs.tolist())]]
        return masked_content_logprob, top_logprobs

def augment_partial_queries_with_contexts(
    queries: Iterable[dict[str, str | None]],
    get_augmented_context: Callable[[dict[str, str | None]], str] = lambda _: "",
) -> Iterable[dict[str, str | None]]:
    for query in queries:
        query["augmented_context"] = get_augmented_context(query)
        yield query


if __name__ == "__main__":
    input_path = "/home/allenwang/pra/MS-KeBAB/tests/data/text_completion/plain_text_items.jsonl"
    docReader = DocumentJsonlReader(Path(input_path))
    docs = docReader.read_items()
    queries = BaseTextCompleter.generate_partial_queries(docs)
    queries = augment_partial_queries_with_contexts(queries)
    phi_annotator = PhiTextCompleter()
    metrics = phi_annotator.complete_text_and_evaluate(queries, verbose=True)

    output_path = os.path.splitext(input_path)[0] + "_tc_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    doc_id = "Invalid"
    results_per_doc = []
    for result in metrics["all_results"]:
        if result["document_id"] != doc_id and doc_id != "Invalid":
            BaseTextCompleter.generate_annotated_doc_html(results_per_doc, os.path.splitext(input_path)[0] + f"_annotated_{doc_id}.html")
            results_per_doc = []
        else:
            results_per_doc.append(result)
        doc_id = result["document_id"]
    if results_per_doc:
        BaseTextCompleter.generate_annotated_doc_html(results_per_doc, os.path.splitext(input_path)[0] + f"_annotated_{doc_id}.html")
        results_per_doc = []
