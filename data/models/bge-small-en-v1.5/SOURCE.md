# bge-small-en-v1.5 tokenizer (vendored)

Only the tokenizer is committed, so offline CI can rebuild and check the chunk
manifest. The model weights (`onnx/model.onnx`) are not committed; `make model-fetch`
downloads them to `.cache/` and checks them against their pin in `config/rag.yaml`.

| | |
|---|---|
| File | `tokenizer.json` |
| Upstream | Hugging Face `BAAI/bge-small-en-v1.5`, path `tokenizer.json` |
| Revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| URL | https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json |
| SHA-256 | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |
| Size | 711,396 bytes |
| Retrieved | 2026-10-12 |
| Modified | no (byte-identical to upstream) |

## License

MIT. The model card at this revision declares `license: mit` and states: "FlagEmbedding
is licensed under the MIT License. The released models can be used for commercial
purposes free of charge." The Hugging Face repository has no license file of its own,
so `LICENSE` here is the MIT license the model card links to, copied verbatim from
https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE at commit
`fd1a2bdf69488ffebe0327999d4400d8c8058a0b` (retrieved 2026-10-12).
