#!/usr/bin/env python3
from src.encoding.leafy_chain_packer import pack_sequences


def main():
    code_tokens = ["def", "foo", "(", ")", ":"]
    leaves = [
        {"head": "foo", "relation": "calls", "tail_tokens": ["bar"]},
        {"head": "foo", "relation": "reads_from", "tail_tokens": ["x"]},
    ]
    ids = pack_sequences(code_tokens, leaves, max_seq_len=32)
    print("Packed length:", len(ids))

if __name__ == "__main__":
    main()
