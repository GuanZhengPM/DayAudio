# Contributing

1. Do not commit real recordings, transcripts, model weights, API keys, local
   paths, or voice embeddings.
2. Add a generated/public fixture and regression test for behavior changes.
3. Keep optional ML imports inside adapters.
4. Record model ID, revision, digest, decode configuration, and hardware in
   benchmark reports.
5. Speedups require the same source set and quality comparison; model
   disagreement is not WER/CER without human reference.
6. Speaker consistency against another model is not DER. Owner behavior needs
   explicit enrollment and positive/negative evaluation.

Run `pytest` and `python -m build` before submitting a pull request.
