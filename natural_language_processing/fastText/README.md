# fastText

## Example use cases

### Compile

```shell
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```

### Word Embedding

```shell
$ ./fasttext skipgram -input data.txt -output model
```

`data.txt`: training file (concate all sentences in the first line)

`model`: model name

### Obtaining word vectors for out-of-vocabulary words

```shell
$ ./fasttext print-word-vectors model.bin < queries.txt > output.txt
```

`queries.txt`: queries (put one word in one line)

`output.txt`: vectors correspond to `queries.txt` line-by-line
