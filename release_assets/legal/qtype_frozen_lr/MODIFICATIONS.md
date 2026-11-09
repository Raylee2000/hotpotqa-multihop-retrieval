# Modification notice — qtype_frozen_lr

Project changes by Yi-Jui Li.

No third-party pretrained weights are included. lr_full.npz and lr_fold0/1.npz contain project-fitted Logistic Regression coefficients/intercepts and fitting metadata on frozen MiniLM embeddings. Two .npy files contain scalar project predictions only. JSON files contain schema, hashes and reload diagnostics. This is the same-condition refit of qtype_v2; original qtype_v2 weights were not saved. The encoder is a separately downloaded dependency, not redistributed in this ZIP.

For binary weights the above is the accompanying prominent modification notice; the binary file bytes are preserved to retain the exact research checkpoint. Tokenizer/config serialization originates from the source run listed in PROVENANCE.json; unmodified upstream files are identified there when their SHA256 matches the pinned base manifest.

Release preparation (2026-11-09): added LICENSE, NOTICE, this modification statement, DATA_SOURCES.md and PROVENANCE.json. No existing archive member was changed; all original parameter and metadata bytes retain their previous SHA256. There was no new training, refit, inference or selection.
