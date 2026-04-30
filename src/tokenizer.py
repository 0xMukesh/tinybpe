import itertools
from collections import Counter


class Tokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def _get_stats(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        return Counter(itertools.pairwise(tokens))

    def _merge_tokens(
        self, tokens: list[int], tokens_to_merge: tuple[int, int], new_token_idx: int
    ) -> list[int]:
        new_tokens: list[int] = []
        i = 0

        while i < len(tokens):
            if (
                tokens[i] == tokens_to_merge[0]
                and i < len(tokens) - 1
                and tokens[i + 1] == tokens_to_merge[1]
            ):
                new_tokens.append(new_token_idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256, (
            "vocab size must be atleast 256 to be able to encode all characters in UTF-8"
        )
        n_merges = vocab_size - 256

        tokens = list(text.encode("utf-8"))

        for i in range(n_merges):
            stats = self._get_stats(tokens)
            tokens_to_merge = max(stats.items(), key=lambda p: p[1])[0]
            token_idx = 256 + i
            tokens = self._merge_tokens(tokens, tokens_to_merge, token_idx)

            self.vocab[token_idx] = (
                self.vocab[tokens_to_merge[0]] + self.vocab[tokens_to_merge[1]]
            )
            self.merges[tokens_to_merge] = token_idx

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))

        while True:
            stats = self._get_stats(tokens)
            if len(stats) == 0:
                break

            to_merge = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if to_merge not in tokens:
                break

            tokens = self._merge_tokens(tokens, to_merge, self.merges[to_merge])

        return tokens

    def decode(self, tokens: list[int]) -> str:
        bytes = b"".join(self.vocab[idx] for idx in tokens)
        return bytes.decode("utf-8", errors="replace")
