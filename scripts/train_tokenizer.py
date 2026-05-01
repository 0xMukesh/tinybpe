import argparse
import cProfile
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.tokenizer import SplitPattern, Tokenizer
from src.utils import find_chunk_boundaries


def build_special_tokens(vocab_size: int) -> dict[str, int]:
    return {"<|endoftext|>": vocab_size}


parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--profile", action="store_true", default=False)
parser.add_argument("--vocab-size", type=int, required=True)
parser.add_argument("--num-chunks", type=int, required=True)
args = parser.parse_args()


def main():
    special_tokens = build_special_tokens(args.vocab_size)
    split_token = list(special_tokens.keys())[0].encode()

    input_file_path = Path(args.input_file)
    with open(input_file_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, args.num_chunks, split_token)
        chunks: list[str] = []

        for start, end in itertools.pairwise(chunk_boundaries):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8"))

    tokenizer = Tokenizer(SplitPattern.GPT4, special_tokens)
    train_start = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        executor.map(
            lambda ch: tokenizer.train(
                ch,
                args.vocab_size,
            ),
        )

    tokenizer._save_state("./data/tokenizer.pkl")
    print(f"took {time.perf_counter() - train_start}s to train bpe tokenizer")


if __name__ == "__main__":
    if args.profile:
        cProfile.run("main()")
    else:
        main()
