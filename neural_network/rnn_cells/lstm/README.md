# LSTM (Long Short-Term Memory)

$$
\begin{align}
y_t, h_t, c_t &= f(x, h_{t-1}, c_{t-1}) \\
x' &= [x_t^T, h_{t-1}^T]^T \\
z &= \tanh(W x') \\
z^i &= \sigma_1(W^i x') \\
z^f &= \sigma_2(W^f x') \\
z' &= \sigma_3(W' x') \\
c_t &= z^f \bullet c^{t-1} + z^i \bullet z \\
h_t &= z' \bullet \tanh(c_t) \\
y_t &= softmax(W^o h_t)
\end{align}
$$

$$
\begin{array}{c}
x_t \in R^n \text{: input vectors at time }t \\
c_{t-i}, h_{t-i} \in R^n \text{: state vectors at time }t-i \\
y_t \in R^k \text{: output vectors at time }t \\
W, W^i, W^f, W' \in R^{n \times 2n}, W^o \in R^{k \times n} \text{: trainable variables} \\
\sigma_i \text{: activation functions}
\end{array}
$$

## Peephole

$$
\begin{array}{c}
x' = [x_t^T, h_{t-1}^T, c_{t-1}^T]^T \\
W, W^i, W^f, W' \in R^{n \times 3n} \text{: trainable variables} \\
\end{array}
$$

## References

* [Prof. Hung-Yi Lee](http://speech.ee.ntu.edu.tw/~tlkagk/index.html)