# Third-Party Licenses and Attributions

This repository's original source code is licensed under the MIT License (see `LICENSE`).

The MIT License does not supersede the licenses or usage conditions of third-party datasets, pretrained models, libraries, or other assets.

## HotpotQA

- Dataset: HotpotQA
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- Source: https://hotpotqa.github.io/
- License: https://creativecommons.org/licenses/by-sa/4.0/

HotpotQA data remains subject to its original license. Users must comply with applicable attribution and ShareAlike requirements when redistributing or adapting covered material.

## Sentence Transformers / MiniLM

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- License: Apache License 2.0
- Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- License: https://www.apache.org/licenses/LICENSE-2.0

The pretrained model and any applicable redistributed components remain subject to their respective licensing terms. Required copyright notices and attribution must be preserved.

## Fine-Tuned Model Artifacts

The six v0.1.0 model ZIPs are distributed under **Apache-2.0**, including the project's fine-tuning, learned parameters and release documentation. Copyright (c) 2026 Yi-Jui Li applies to the project contributions, not to upstream work. This explicit model-artifact license is separate from the source-code MIT license. The archive revision is `license-metadata-1` in `configs/models.json`.

| Asset | Audited contents and origin | Redistribution license |
|---|---|---|
| `hotpotqa-general-encoder-v1.zip` | MiniLM fine-tuned with project hard negatives; `outputs/training_v1/hardneg_v1/snapshot_step01869`; includes saved tokenizer/configuration | Apache-2.0; upstream attribution and modification notice included |
| `hotpotqa-csh-v1.zip` | Conditional second-hop MiniLM; `outputs/csh_v1/csh_main_v1/snapshot_step01500`; includes saved tokenizer/configuration | Apache-2.0; upstream attribution and modification notice included |
| `hotpotqa-sf-selector-fold0-v1.zip` | Fine-tuned MiniLM and project sentence-classification head; `outputs/sf_selector_v2/sf_cf_v1/fold0/final` | Apache-2.0; upstream attribution and modification notice included |
| `hotpotqa-sf-selector-fold1-v1.zip` | Fine-tuned MiniLM and project sentence-classification head; `outputs/sf_selector_v2/sf_cf_v1/fold1/final` | Apache-2.0; upstream attribution and modification notice included |
| `hotpotqa-qtype-frozen-lr-v1.zip` | Three project-fitted LR parameter files, two scalar-probability arrays, schemas and reload/provenance JSON; `outputs/qtype_artifacts_v1/qtype_v2_frozen_lr` | Apache-2.0 for project artifacts; **no pretrained model, tokenizer or third-party library code included** |
| `hotpotqa-r5-full-v1.zip` | Numeric tree arrays and feature schema exported from the project's 200-tree LambdaMART model; `outputs/path_reranker_refit_v1/general_csh/R5_lambdamart__full` | Apache-2.0 for project artifacts; **no pretrained model, tokenizer, scikit-learn code or joblib included** |

The `outputs/...` paths above identify read-only research sources used to check the release; those directories are not required or distributed with the public installation. Every one of the 45 pre-existing archive files was checked against its research checkpoint or existing converted source. The original archive member bytes, including all weights, remain unchanged. SF heads were additionally checked tensor-for-tensor against their original restricted-load `head.pt` state dictionaries. The R5 NPZ records the original joblib SHA256 (`a8d9dfb0d745261620dda78af6fb91e137eb6a3133c33861d28566acdc7a8da5`); that source hash was checked without executing pickle.

### License and provenance supplied with each ZIP

Each ZIP contains `LICENSE` (complete Apache-2.0 terms and appendix), `NOTICE`, `MODIFICATIONS.md`, `DATA_SOURCES.md`, and `PROVENANCE.json`. The provenance records every original file's SHA256, source path, and the pre-notice archive hash. `configs/models.json` includes the added documents in the exact extraction allowlist and per-file SHA256 checks; `release_assets/SHA256SUMS` identifies the updated archives.

The four Encoder/SF ZIPs additionally retain the **unmodified upstream model card** in `UPSTREAM_MODEL_CARD.md`, verified against the pinned base-model manifest. The source is:

- Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md
- Pinned file tree: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
- License text: https://www.apache.org/licenses/LICENSE-2.0.txt

The upstream card declares Apache-2.0 and identifies `nreimers/MiniLM-L6-H384-uncased` as the base model. That model is MIT-licensed and identifies `microsoft/MiniLM-L12-H384-uncased` as its source. The four Encoder/SF ZIPs therefore also include `MINILM_MIT_LICENSE.txt`, preserving the complete upstream MIT terms and `Copyright (c) Microsoft Corporation`; Apache-2.0 does not replace those upstream notices. Sources checked on 2026-11-09:

- https://huggingface.co/nreimers/MiniLM-L6-H384-uncased
- https://huggingface.co/microsoft/MiniLM-L12-H384-uncased
- https://raw.githubusercontent.com/microsoft/unilm/master/LICENSE

All attributions in the pinned Sentence Transformers model card are retained. No separate `NOTICE` file is listed at that pinned repository root; the supplied `NOTICE` is project attribution, not an invented upstream NOTICE. `MODIFICATIONS.md` prominently identifies fine-tuned binary weights, the project-trained SF heads, configuration/tokenizer exports, and release-only additions. Upstream rights are not reassigned to the project, and upstream endorsement is not claimed.

### HotpotQA training-data attribution and scope

Attribution: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov and Christopher D. Manning. 2018. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018.

The project used fixed HotpotQA splits, supporting-fact/type labels, mined negatives and numeric pair features. The Hugging Face dataset revision is `1908d6afbbead072334abe2965f91bd2709910ab`, as recorded in `data/manifests/release_data_lock.json`.

**None of these six ZIPs bundles original HotpotQA questions, answers, context article bodies, supporting-fact records, training embeddings or candidate feature tables.** QType's two `.npy` files contain project-generated scalar probabilities only; its `.npz` files contain LR coefficients/intercepts and numerical fitting metadata. R5 contains tree arrays, thresholds, leaf values, feature names and scalar/schema metadata. Both formats were inspected with `allow_pickle=False`; neither ZIP distributes the training libraries. The metadata identifies the data source but does not redistribute dataset text.

CC BY-SA 4.0 continues to govern HotpotQA data. Apache-2.0 is the license for the supplied project model artifacts; this is not a relicensing of the dataset, nor a claim that merely training on CC BY-SA data automatically applies ShareAlike to every learned coefficient. Dataset attribution and the data-license link are carried in every ZIP's `DATA_SOURCES.md` and `NOTICE`. Any separately distributed HotpotQA data must retain its own applicable attribution and ShareAlike conditions.

The release attachments are exactly the six ZIPs listed in `configs/models.json` and `release_assets/SHA256SUMS`; earlier pre-notice archives are not release attachments. The license and provenance documents contained in each ZIP are also tracked in the repository under `release_assets/legal/<model>/` (byte-identical to the archive members, SHA256 in `configs/models.json`). `tools/maintainer/package_assets.py` packs them into the ZIPs, and `scripts/export_trained_models.py` uses them for models retrained with `python run.py train`.

## Python Dependencies

Third-party Python packages retain their own licenses. The dependencies are listed in `environment.yml` and pinned in `requirements-lock.txt` (`pyproject.toml` declares no runtime dependencies); see the installed package distributions for dependency-specific license information.
