# Modification notice — general_encoder

Project changes by Yi-Jui Li.

model.safetensors: full-parameter contrastive fine-tuning of the pinned MiniLM with HotpotQA hard negatives; saved checkpoint step 1869. The JSON/tokenizer/pooling files were saved by Sentence Transformers; PROVENANCE.json records their exact origins.

For binary weights the above is the accompanying prominent modification notice; the binary file bytes are preserved to retain the exact research checkpoint. Tokenizer/config serialization originates from the source run listed in PROVENANCE.json; unmodified upstream files are identified there when their SHA256 matches the pinned base manifest.

Release preparation (2026-11-09): added LICENSE, NOTICE, this modification statement, DATA_SOURCES.md and PROVENANCE.json, plus the unmodified upstream model card. No existing archive member was changed; all original parameter and metadata bytes retain their previous SHA256. There was no new training, refit, inference or selection.

The release also preserves the MiniLM ancestor MIT license and Microsoft copyright notice in MINILM_MIT_LICENSE.txt.
