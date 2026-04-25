# SCALAR Part-of-Speech Tagger for Identifiers

**SCALAR** is a part-of-speech tagger for source code identifiers. It supports two model types:

- **DistilBERT-based model with CRF layer** (Recommended: faster, more accurate)
- Legacy Gradient Boosting model (for compatibility)

---

## Installation

Make sure you have `python3.12` installed. Then:

```bash
git clone https://github.com/SCANL/scanl_tagger.git
cd scanl_tagger
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

You can run SCALAR in multiple ways:

### CLI (with DistilBERT or GradientBoosting model)

```bash
python main --mode run --model_type lm_based         # DistilBERT (recommended)
python main --mode run --model_type tree_based       # Legacy model
```

Then query like:

```
http://127.0.0.1:8080/GetValue/FUNCTION
```

Supports context types:
- FUNCTION
- CLASS
- ATTRIBUTE
- DECLARATION
- PARAMETER

### POSTバッチサーバ

複数の識別子をまとめてタグ付けするための POST ベースのバッチサーバも起動できます:

```bash
./venv/bin/python serve_post_batch.py --port 8091 --protocol http --worker-processes 10 --threads 10
```

単一 GPU で動かす場合は、`--worker-processes` を増やすと同じ GPU にモデルが複数ロードされます。まず `--worker-processes 1` から試すのがおすすめです。

複数 GPU を使う場合は、同じサーバの中で GPU を直接指定できます:

```bash
./venv/bin/python serve_post_batch.py \
  --port 8091 \
  --protocol http \
  --gpus 0,1 \
  --worker-processes 4 \
  --max-batch-size 10
```

`--worker-processes` は全 GPU を合わせた worker 数で、上の例では `cuda:0` と `cuda:1` に worker が分散されます。利用可能な GPU をすべて使いたい場合は `--gpus all` も指定できます。

`--max-batch-size` は、cache miss した識別子を 1 回の model forward に何件までまとめるかを制御します。デフォルトは `10` です。GPU メモリに余裕がある場合だけ、`32` や `64` などへ上げてください。

起動後は、次のエンドポイントに POST リクエストを送ります:

```text
http://127.0.0.1:8091/tag
```

リクエストボディの例:

```json
{
  "functions": ["get_user_name", "validate_input"],
  "parameters": ["user_id", "max_count"],
  "variables": ["tmp_result", "cache_map"]
}
```

キャッシュに関する補足:

- POST バッチサーバは、メインの HTTP サーバプロセス内で in-memory の LRU cache を使用します。
- キャッシュキーには `identifier_name`, `context`, `system_name`, `language`, `type` が含まれます。
- キャッシュはプロセス間で共有されません。worker process 側は、この LRU cache の共有コピーを持ちません。
- `--cache-size` は、メインプロセス内に保持するキャッシュ件数を制御します。

### 負荷テスト

数字と記号の prefix / suffix を識別子に付けて LRU cache を避けながら、繰り返し POST リクエストを送るには次を使います:

```bash
scripts/load_test_post_batch.py \
  --url http://127.0.0.1:8091/tag \
  --requests 10000 \
  --concurrency 16
```

---

## Training

You can retrain either model (default parameters are currently hardcoded):

```bash
python main --mode train --model_type lm_based
python main --mode train --model_type tree_based
```

---

## Evaluation Results

### DistilBERT (LM-Based Model) — Recommended

| Metric                   | Score   |
|--------------------------|---------|
| **Macro F1**             | 0.9032  |
| **Token Accuracy**       | 0.9223  |
| **Identifier Accuracy**  | 0.8291  |

| Label | Precision | Recall | F1    | Support |
|-------|-----------|--------|-------|---------|
| CJ    | 0.88      | 0.88   | 0.88  | 8       |
| D     | 0.98      | 0.96   | 0.97  | 52      |
| DT    | 0.95      | 0.93   | 0.94  | 45      |
| N     | 0.94      | 0.94   | 0.94  | 418     |
| NM    | 0.91      | 0.93   | 0.92  | 440     |
| NPL   | 0.97      | 0.97   | 0.97  | 79      |
| P     | 0.94      | 0.92   | 0.93  | 79      |
| PRE   | 0.79      | 0.79   | 0.79  | 68      |
| V     | 0.89      | 0.84   | 0.86  | 110     |
| VM    | 0.79      | 0.85   | 0.81  | 13      |

**Inference Performance:**
- Identifiers/sec: 225.8

---

### Gradient Boost Model (Legacy)

| Metric               | Score     |
|----------------------|-----------|
| Accuracy             | 0.8216    |
| Balanced Accuracy    | 0.9160    |
| Weighted Recall      | 0.8216    |
| Weighted Precision   | 0.8245    |
| Weighted F1          | 0.8220    |
| Inference Time       | 249.05s   |

**Inference Performance:**
- Identifiers/sec: 8.6

---

## Supported Tagset

| Tag   | Meaning                            | Examples                       |
|-------|------------------------------------|--------------------------------|
| N     | Noun                               | `user`, `Data`, `Array`        |
| DT    | Determiner                         | `this`, `that`, `those`        |
| CJ    | Conjunction                        | `and`, `or`, `but`             |
| P     | Preposition                        | `with`, `for`, `in`            |
| NPL   | Plural Noun                        | `elements`, `indices`          |
| NM    | Noun Modifier (adjective-like)     | `max`, `total`, `employee`     |
| V     | Verb                               | `get`, `set`, `delete`         |
| VM    | Verb Modifier (adverb-like)        | `quickly`, `deeply`            |
| D     | Digit                              | `1`, `2`, `10`, `0xAF`         |
| PRE   | Preamble / Prefix                  | `m`, `b`, `GL`, `p`            |

---

## Docker Support (Legacy only)

For the legacy server, you can also use Docker:

```bash
docker compose pull
docker compose up
```

---

## Notes

- **Kebab case** is not supported (e.g., `do-something-cool`).
- Feature and position tokens (e.g., `@pos_0`) are inserted automatically.
- Internally uses [WordNet](https://wordnet.princeton.edu/) for lexical features.
- Input must be parsed into identifier tokens. We recommend [srcML](https://www.srcml.org/) but any AST-based parser works.

---

## Citations

Please cite:

```
@inproceedings{newman2025scalar,
  author    = {Christian Newman and Brandon Scholten and Sophia Testa and others},
  title     = {SCALAR: A Part-of-speech Tagger for Identifiers},
  booktitle = {ICPC Tool Demonstrations Track},
  year      = {2025}
}

@article{newman2021ensemble,
  title={An Ensemble Approach for Annotating Source Code Identifiers with Part-of-speech Tags},
  author={Newman, Christian and Decker, Michael and AlSuhaibani, Reem and others},
  journal={IEEE Transactions on Software Engineering},
  year={2021},
  doi={10.1109/TSE.2021.3098242}
}
```

---

## Training Data

You can find the most recent SCALAR training dataset [here](https://github.com/SCANL/scanl_tagger/blob/master/input/tagger_data.tsv)

---

## More from SCANL

- [SCANL Website](https://www.scanl.org/)
- [Identifier Name Structure Catalogue](https://github.com/SCANL/identifier_name_structure_catalogue)

---

## Trouble?

Please [open an issue](https://github.com/SCANL/scanl_tagger/issues) if you encounter problems!
