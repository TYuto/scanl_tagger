import json
import os

import nltk
from spiral import ronin

from src.lm_based_tagger.distilbert_tagger import DistilBertTagger
from src.tree_based_tagger.download_code2vec_vectors import download_files


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_REMOTE_MODEL = "sourceslicer/scalar_lm_best"


class WordList:
    def __init__(self, path):
        self.path = path
        self.words = set()

    def load(self):
        if not self.path or not os.path.isfile(self.path):
            return

        with open(self.path) as handle:
            for line in handle:
                comma_index = line.find(",")
                if comma_index == -1:
                    token = line.strip()
                else:
                    token = line[:comma_index].strip()

                if token:
                    self.words.add(token)

    def find(self, item):
        return item in self.words


def load_serve_config():
    with open(os.path.join(REPO_ROOT, "serve.json")) as handle:
        return json.load(handle)


def build_lm_model_path(local=False, model=None):
    if model:
        return model
    if local:
        return os.path.join(REPO_ROOT, "output", "best_model")
    return DEFAULT_REMOTE_MODEL


class LMRuntime:
    def __init__(self, model_path, local=False, device=None, require_gpu=False, words_path=None):
        download_files()
        self.english_words = set(w.lower() for w in nltk.corpus.words.words())
        self.word_list = WordList(words_path or self._resolve_words_path())
        self.word_list.load()
        self.tagger = DistilBertTagger(
            model_path=model_path,
            local=local,
            device=device,
            require_gpu=require_gpu,
        )
        self.model_device = str(self.tagger.device)

    def _resolve_words_path(self):
        config = load_serve_config()
        return config.get("words", "")

    def dictionary_lookup(self, word):
        if word.lower() in self.english_words:
            return "DW"
        if self.word_list.find(word):
            return "AW"
        if word.isnumeric():
            return "DD"
        return "UC"

    def build_result(self, words, tags):
        result = {"words": []}

        for index, word in enumerate(words):
            tag = tags[index] if index < len(tags) else None
            result["words"].append({
                word: {
                    "tag": tag,
                    "dictionary": self.dictionary_lookup(word),
                }
            })

        return result

    def tag_identifier_result(self, identifier_name, identifier_context, system_name="", programming_language="", data_type=""):
        words = ronin.split(identifier_name)
        tags = self.tagger.tag_identifier(
            tokens=words,
            context=identifier_context,
            type_str=data_type,
            language=programming_language,
            system_name=system_name,
        )
        return self.build_result(words, tags)

    def tag_identifier_batch_results(self, request_payloads):
        prepared = []
        for payload in request_payloads:
            words = ronin.split(payload["identifier_name"])
            prepared.append({
                "words": words,
                "context": payload["identifier_context"],
                "type_str": payload.get("data_type", ""),
                "language": payload.get("programming_language", ""),
                "system_name": payload.get("system_name", ""),
            })

        batched_tags = self.tagger.tag_identifiers([
            {
                "tokens": item["words"],
                "context": item["context"],
                "type_str": item["type_str"],
                "language": item["language"],
                "system_name": item["system_name"],
            }
            for item in prepared
        ])

        results = []
        for item, tags in zip(prepared, batched_tags):
            results.append(self.build_result(item["words"], tags))

        return results
