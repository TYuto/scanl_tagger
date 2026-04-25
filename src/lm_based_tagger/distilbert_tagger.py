import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from .distilbert_crf import DistilBertCRFForTokenClassification 
from .distilbert_preprocessing import *

class DistilBertTagger:
    """
    A lightweight wrapper around a DistilBERT+CRF or DistilBERT-only model for tagging identifier tokens
    with part-of-speech-like grammar labels (e.g., V, NM, N, etc.).

    Automatically handles:
    - Tokenization (with custom feature and position tokens)
    - Running the model
    - Post-processing the raw logits or CRF predictions
    - Aligning subword tokens back to word-level predictions
    """
    def __init__(self, model_path: str, local: bool = False, device: str | None = None, require_gpu: bool = False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        if require_gpu and self.device.type != "cuda":
            raise RuntimeError("GPU execution was requested, but CUDA is not available.")

        # Load tokenizer from local directory or remote HuggingFace path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=local)

        # Try loading CRF-enhanced model; fallback to plain classifier if not available
        try:
            self.model = DistilBertCRFForTokenClassification.from_pretrained(model_path, local=local)
        except Exception:
            self.model = DistilBertForTokenClassification.from_pretrained(model_path, local_files_only=local)

        self.model.to(self.device)
        # disable dropout, etc. for inference
        self.model.eval()
        
        # map label IDs to strings
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def _build_input_tokens(self, tokens, context, type_str, language, system_name):
        row = {
            "CONTEXT": context,
            "SYSTEM_NAME": system_name,
            "TYPE": type_str,
            "LANGUAGE": language
        }

        feature_tokens = get_feature_tokens(row, tokens)

        length = len(tokens)
        pos_tokens = ["@pos_2"] if length == 1 else ["@pos_0"] + ["@pos_1"] * (length - 2) + ["@pos_2"]
        tokens_with_pos = [val for pair in zip(pos_tokens, tokens) for val in pair]
        return feature_tokens + tokens_with_pos

    def _decode_output(self, out):
        if isinstance(out, dict) and "predictions" in out:
            return out["predictions"]

        if hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        return torch.argmax(logits, dim=-1).tolist()

    def _align_labels(self, word_ids, labels_per_token):
        pred_labels = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx < NUMBER_OF_FEATURES:
                continue
            if (word_idx - NUMBER_OF_FEATURES) % 2 == 0:
                continue
            if word_idx == previous_word_idx:
                continue

            label_idx = idx - 1
            if label_idx < len(labels_per_token):
                pred_labels.append(labels_per_token[label_idx])
            previous_word_idx = word_idx

        return [self.id2label[i] for i in pred_labels]

    def tag_identifier(self, tokens, context, type_str, language, system_name):
        """
        Tag a split identifier using the model, returning a sequence of grammar pattern labels (e.g., ["V", "NM", "N"]).

        Steps:
        1) Build full input token list:
              [feature tokens] + [@pos_0, w1, @pos_1, w2, ..., @pos_2, wn]
        2) Tokenize using HuggingFace tokenizer with is_split_into_words=True
        3) Run the model forward pass (handles CRF or logits automatically)
        4) Use word_ids() to align predictions back to full words
              - Skip special tokens (None)
              - Skip feature tokens (index < NUMBER_OF_FEATURES)
              - Use only the *second* token in each [@pos_X, word] pair (the word)
              - Skip repeated subword tokens (only use the first subtoken per word)
        5) Return a list of string labels corresponding to the original identifier tokens.

        Returns:
            List[str]: a list of grammar tags (e.g., ['V', 'NM', 'N']) aligned to `tokens`
        """
        return self.tag_identifiers([{
            "tokens": tokens,
            "context": context,
            "type_str": type_str,
            "language": language,
            "system_name": system_name,
        }])[0]

    def tag_identifiers(self, items):
        input_batches = [
            self._build_input_tokens(
                tokens=item["tokens"],
                context=item["context"],
                type_str=item.get("type_str", ""),
                language=item.get("language", ""),
                system_name=item.get("system_name", ""),
            )
            for item in items
        ]

        encoded = self.tokenizer(
            input_batches,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        model_inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

        with torch.inference_mode():
            out = self.model(**model_inputs)

        labels_per_batch = self._decode_output(out)
        if items and labels_per_batch and isinstance(labels_per_batch[0], int):
            labels_per_batch = [labels_per_batch]

        predictions = []
        for batch_index, labels_per_token in enumerate(labels_per_batch):
            word_ids = encoded.word_ids(batch_index=batch_index)
            predictions.append(self._align_labels(word_ids, labels_per_token))

        return predictions
