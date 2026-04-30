from pathlib import Path

from src.tokenizer import Tokenizer


def main():
    text = Path("./data/content.txt").read_text().strip()
    tokenizer = Tokenizer()
    tokenizer.train(text, 258)
    output = tokenizer.encode("hello, world")
    print(output)


if __name__ == "__main__":
    main()
