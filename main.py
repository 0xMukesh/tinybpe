import argparse
import cProfile
import os
import time
from concurrent.futures import ProcessPoolExecutor

from src.tokenizer import SplitPattern, Tokenizer, _iter_chunks
from src.utils import find_chunk_boundaries


def build_special_tokens(vocab_size: int):
    return {"<|endoftext|>": vocab_size}


def train_tokenizer(args: argparse.Namespace, tokenizer: Tokenizer, split_token: bytes):
    assert args.vocab_size is not None, "--vocab-size is required for training"
    tokenizer.train(
        args.input_file,
        args.vocab_size,
        split_token,
        args.num_workers,
        args.num_chunks_per_worker,
    )
    if args.state_file:
        tokenizer._save_state(args.state_file)

    text = "hello, world <|endoftext|>"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"encoding of {text}: {encoded}")
    print(f"decoding of {encoded}: {decoded}")
    print(f"are equal: {decoded == text}")

    max_len_token = str(max(tokenizer.vocab.items(), key=lambda x: len(x[1]))[1])
    print(f"token with maximum length: {max_len_token}")


def benchmark_tokenizer(
    args: argparse.Namespace, tokenizer: Tokenizer, split_token: bytes
):
    assert args.state_file is not None, "--state-file is required for benchmarking"
    tokenizer._load_state(args.state_file)

    with open(args.input_file, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_boundaries = find_chunk_boundaries(
            f, args.num_workers * args.num_chunks_per_worker, split_token
        )

        with ProcessPoolExecutor(max_workers=args.num_workers) as p:
            start = time.perf_counter()
            p.map(tokenizer.encode, _iter_chunks(f, chunk_boundaries), chunksize=2)
            end = time.perf_counter()
            throughput = file_size / (end - start)

    print(f"throughput: {throughput} bytes/sec")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["train", "benchmark"])
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--num-chunks-per-worker", type=int, default=4)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--state-file", type=str)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()

    profiler = cProfile.Profile()
    if args.profile:
        profiler.enable()

    special_tokens = build_special_tokens(args.vocab_size)
    split_token = list(special_tokens.keys())[0].encode()

    tokenizer = Tokenizer(SplitPattern.GPT4, special_tokens)

    if args.action == "train":
        train_tokenizer(args, tokenizer, split_token)
    elif args.action == "benchmark":
        benchmark_tokenizer(args, tokenizer, split_token)

    if args.profile:
        profiler.print_stats(sort="time")


if __name__ == "__main__":
    main()
