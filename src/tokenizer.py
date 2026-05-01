import itertools
import pickle
from collections import Counter
from enum import Enum

import regex as re

DEFAULT_TOKENIZER_STATE_PATH = "tokenizer.pkl"
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class SplitPattern(Enum):
    GPT2 = "gpt2"
    GPT4 = "gpt4"


def split_pattern_to_regex(split_pattern: SplitPattern) -> str:
    if split_pattern == SplitPattern.GPT2:
        return GPT2_SPLIT_PATTERN
    elif split_pattern == SplitPattern.GPT4:
        return GPT4_SPLIT_PATTERN


class Tokenizer:
    def __init__(self, split_pattern: SplitPattern, special_tokens: dict[str, int]):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

        self.pattern = re.compile(
            split_pattern_to_regex(split_pattern), flags=re.IGNORECASE
        )
        self.special_pattern = (
            "(" + "|".join(str(re.escape(k)) for k in special_tokens) + ")"
        )

    @staticmethod
    def _get_stats(tokens: list[int]) -> Counter[tuple[int, int]]:
        return Counter(itertools.pairwise(tokens))

    def _split_on_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]

        pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        parts = re.split(pattern, text)
        return parts

    def _merge_tokens_and_update_stats(
        self,
        tokens: list[int],
        tokens_to_merge: tuple[int, int],
        new_token_idx: int,
        stats: Counter,
    ) -> list[int]:
        i, j = 0, 0
        n = len(tokens)
        new_tokens: list[int] = [0] * n

        while i < n:
            if (
                tokens[i] == tokens_to_merge[0]
                and i < n - 1
                and tokens[i + 1] == tokens_to_merge[1]
            ):
                if i > 0:
                    stats[(tokens[i - 1], tokens[i])] -= 1
                    stats[(tokens[i - 1], new_token_idx)] += 1
                if i < n - 2:
                    stats[(tokens[i + 1], tokens[i + 2])] -= 1
                    stats[(new_token_idx, tokens[i + 2])] += 1

                new_tokens[j] = new_token_idx
                j += 1
                i += 2
            else:
                new_tokens[j] = tokens[i]
                j += 1
                i += 1

        return new_tokens[:j]

    def _save_state(self, path: str = DEFAULT_TOKENIZER_STATE_PATH):
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab": self.vocab}, f)

    def _load_state(self, path: str = DEFAULT_TOKENIZER_STATE_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.merges = data["merges"]
            self.vocab = data["vocab"]

    def train(
        self,
        text: str,
        vocab_size: int,
        save_state: bool = False,
        output_path: str = DEFAULT_TOKENIZER_STATE_PATH,
    ):
        assert vocab_size >= 256, (
            "vocab size must be atleast 256 to be able to encode all characters in UTF-8"
        )
        n_merges = vocab_size - 256

        chunks: list[list[int]] = []

        for part in self._split_on_special_tokens(text):
            if part in self.special_tokens:
                continue

            matches = re.findall(self.pattern, part)
            chunks.extend(list(m.encode("utf-8")) for m in matches)

        stats: Counter[tuple[int, int]] = Counter()
        for chunk in chunks:
            stats += self._get_stats(chunk)

        for i in range(n_merges):
            if not stats:
                break

            tokens_to_merge = max(stats, key=stats.__getitem__)
            token_idx = 256 + i

            chunks = [
                self._merge_tokens_and_update_stats(
                    chunk, tokens_to_merge, token_idx, stats
                )
                for chunk in chunks
            ]

            del stats[tokens_to_merge]
            self.vocab[token_idx] = (
                self.vocab[tokens_to_merge[0]] + self.vocab[tokens_to_merge[1]]
            )
            self.merges[tokens_to_merge] = token_idx

        if save_state:
            self._save_state(output_path)

    def _encode_bytes(self, text_bytes: bytes) -> list[int]:
        tokens = list(text_bytes)
        stats = self._get_stats(tokens)

        while True:
            if len(stats) == 0:
                break

            to_merge = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if to_merge not in tokens:
                break

            tokens = self._merge_tokens_and_update_stats(
                tokens, to_merge, self.merges[to_merge], stats
            )

        return tokens

    def encode(self, text: str) -> list[int]:
        parts = self._split_on_special_tokens(text)
        tokens: list[int] = []

        for p in parts:
            if p in self.special_tokens:
                tokens.append(self.special_tokens[p])
            else:
                chunks: list[str] = re.findall(self.pattern, p)
                for ch in chunks:
                    tokens.extend(self._encode_bytes(ch.encode("utf-8")))

        return tokens

    def decode(self, tokens: list[int]) -> str:
        part_bytes: list[bytes] = []

        for idx in tokens:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")

        return b"".join(part_bytes).decode("utf-8")
