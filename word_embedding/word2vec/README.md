# Word2vec

The basic concept behind word2vec is that we believe there must be some
relationship between a word and words around it and try to find a vector
representation for a word based on this relationship.

CBOW and skip-gram are two neural networks that consist of an embedding matrix,
which is what we want in the end, and some projection layers.

## CBOW (Continuous Bag-of-Words Model)

Consider a part of a sentence `a b c d e`, we can generate a training example
(`a b d e`, `c`) (window size = 2)

- context vector: one-hot vector multiplied by an embedding matrix
- input: the sum of context vectors of words around `c` (context vectors of `a b d e`)
- output: `c`'s one-hot vector

## Continuous Skip-gram Model

Consider a part of a sentence `a b c d e`, we can generate training examples
(`c`, `a`), (`c`, `b`), (`c`, `d`), (`c`, `e`) (window size = 2)

- context vector: one-hot vector multiplied by an embedding matrix
- input: the context vector of `c`
- output: the corresponding context vector (`a`, `b`, `d`, or `e`)

## Hierarchical Softmax

Significantly reduce the huge computation of the original softmax

- Build a Huffman Tree based on the training corpus
- For each internal node, we need to do a binary classification problem to
decide we should go to the right or left child node and our target is to go
through the right path from the root to the word

Hierarchical softmax performs better for **infrequent** words.

## Negative Sampling

sampled softmax loss on part of negative labels to reduce training time
(probability of a word to be selected will be decided by its frequency)

Negative sampling performs better for **frequent** words.

## References

* [Mikolov et al. 2013](https://arxiv.org/abs/1301.3781)
* [skip-gram by Google](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py)
* [Hierarchical Softmax](http://www.cs.toronto.edu/~fritz/absps/andriytree.pdf)
