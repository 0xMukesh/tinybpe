# tinybpe

a minimal implementation of [byte-pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) with parallel pretokenization, inverse index to find which sequences were affected and priority queues for finding which tokens is to be merged next

### results

i've mainly focused on improving the performance of the tokenizer algorithmatically instead of trying to write C/C++ binding to learn about the algorithmic optimizations

- takes around ~30s (including pretokenization step) to train BPE tokenizer over ~20MB corpus (tinystories validation set) with 8 workers running parallel
- during encoding, tokenizer has a throughput of 232 MB/sec
