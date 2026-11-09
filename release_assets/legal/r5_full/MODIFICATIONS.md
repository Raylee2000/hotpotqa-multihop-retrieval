# Modification notice — r5_full

Project changes by Yi-Jui Li.

No pretrained encoder, tokenizer, third-party executable code or dataset records are included. r5_trees.npz contains project-trained LambdaMART tree arrays and scalar/schema metadata (200 depth-4 trees); feature_schema.json lists 27 project features. Arrays were exported from the project model.joblib using the existing conversion pipeline; the joblib itself and scikit-learn code are not distributed.

For binary weights the above is the accompanying prominent modification notice; the binary file bytes are preserved to retain the exact research checkpoint. Tokenizer/config serialization originates from the source run listed in PROVENANCE.json; unmodified upstream files are identified there when their SHA256 matches the pinned base manifest.

Release preparation (2026-11-09): added LICENSE, NOTICE, this modification statement, DATA_SOURCES.md and PROVENANCE.json. No existing archive member was changed; all original parameter and metadata bytes retain their previous SHA256. There was no new training, refit, inference or selection.
