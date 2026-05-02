import argparse
import cProfile

from src.tokenizer import SplitPattern, Tokenizer


def build_special_tokens(vocab_size: int):
    return {"<|endoftext|>": vocab_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()

    profiler = cProfile.Profile()
    if args.profile:
        profiler.enable()

    special_tokens = build_special_tokens(args.vocab_size)
    split_token = list(special_tokens.keys())[0].encode()

    tokenizer = Tokenizer(SplitPattern.GPT4, special_tokens)
    tokenizer.train(args.input_file, args.vocab_size, split_token, args.num_workers)
    tokenizer._save_state("./data/tokenizer.pkl")

    text = "hello, world <|endoftext|>"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"encoding of {text}: {encoded}")
    print(f"decoding of {encoded}: {decoded}")
    print(f"are equal: {decoded == text}")

    max_len_token = str(max(tokenizer.vocab.items(), key=lambda x: len(x[1]))[1])
    print(f"token with maximum length: {max_len_token}")

    if args.profile:
        profiler.print_stats(sort="time")


if __name__ == "__main__":
    main()
