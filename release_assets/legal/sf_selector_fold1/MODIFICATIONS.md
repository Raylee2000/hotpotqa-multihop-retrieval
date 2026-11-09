# Modification notice — sf_selector_fold1

Project changes by Yi-Jui Li.

model.safetensors: MiniLM fine-tuned as a question/sentence supporting-fact selector on cross-fit fold 1. head.safetensors: project-trained linear classification head, converted from head.pt; tensor values and dtypes were verified identical. Config/tokenizer files were saved by the training/export pipeline.

For binary weights the above is the accompanying prominent modification notice; the binary file bytes are preserved to retain the exact research checkpoint. Tokenizer/config serialization originates from the source run listed in PROVENANCE.json; unmodified upstream files are identified there when their SHA256 matches the pinned base manifest.

Release preparation (2026-11-09): added LICENSE, NOTICE, this modification statement, DATA_SOURCES.md and PROVENANCE.json, plus the unmodified upstream model card. No existing archive member was changed; all original parameter and metadata bytes retain their previous SHA256. There was no new training, refit, inference or selection.

The release also preserves the MiniLM ancestor MIT license and Microsoft copyright notice in MINILM_MIT_LICENSE.txt.
