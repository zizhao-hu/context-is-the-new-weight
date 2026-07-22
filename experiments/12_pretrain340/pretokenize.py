"""Pretokenize FineWeb parquet shards -> flat uint16 memmap for 340M pretraining.

TinyLlama tokenizer (Llama-2 32k vocab, the fla 340M convention); EOS between documents.
Output: /scratch1/zizhaoh/fineweb_tok.bin (uint16) + fineweb_tok.json meta.
"""
import glob, json, os
import numpy as np

os.environ.setdefault("HF_HOME", "/scratch1/zizhaoh/.cache/huggingface")
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = "/scratch1/zizhaoh/fineweb_tok.bin"
META = "/scratch1/zizhaoh/fineweb_tok.json"
SHARDS = sorted(glob.glob("/scratch1/zizhaoh/fineweb/sample/100BT/*.parquet"))
assert SHARDS, "no parquet shards found"
print("shards:", len(SHARDS), flush=True)

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
EOS = tok.eos_token_id
assert tok.vocab_size <= 65535

ds = load_dataset("parquet", data_files=SHARDS, split="train")
print("docs:", len(ds), flush=True)

def enc(batch):
    ids = tok(batch["text"], add_special_tokens=False)["input_ids"]
    return {"ids": [x + [EOS] for x in ids], "n": [len(x) + 1 for x in ids]}

ds = ds.map(enc, batched=True, batch_size=1000, num_proc=32,
            remove_columns=ds.column_names, desc="tokenize")
total = int(np.sum(ds["n"], dtype=np.int64))
print("total tokens:", total, flush=True)

mm = np.memmap(OUT, dtype=np.uint16, mode="w+", shape=(total,))
pos = 0
for row in ds:
    x = np.asarray(row["ids"], dtype=np.uint16)
    mm[pos:pos + len(x)] = x; pos += len(x)
mm.flush()
json.dump({"tokens": total, "vocab": tok.vocab_size + len(tok.added_tokens_decoder),
           "tokenizer": "TinyLlama/TinyLlama_v1.1", "eos": EOS}, open(META, "w"))
print("PRETOK_DONE tokens=%d file=%s" % (total, OUT), flush=True)
