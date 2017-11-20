# RNN

<!---
\begin{align*}
y_t, h_t &= f(x_t, h_{t-1}) \\
h_t &= \sigma_1(W^i x_t + W^h h_{t-1}) \\
y_t &= softmax(W^o h_t)
\end{align*}
\begin{array}{c}
x_t \in R^n \text{: input vectors at time }t \\
h_{t-i} \in R^n \text{: state vectors at time }t-i \\
y_t \in R^k \text{: output vectors at time }t \\
W^i, W^h \in R^{n \times n}, W^o \in R^{k \times n} \text{: trainable variables} \\
\sigma_i \text{: activation functions}
\end{array}
-->
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cbegin%7Balign*%7D%20y_t%2C%20h_t%20%26%3D%20f%28x_t%2C%20h_%7Bt-1%7D%29%20%5C%5C%20h_t%20%26%3D%20%5Csigma_1%28W%5Ei%20x_t%20&plus;%20W%5Eh%20h_%7Bt-1%7D%29%20%5C%5C%20y_t%20%26%3D%20softmax%28W%5Eo%20h_t%29%20%5Cend%7Balign*%7D">
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cbegin%7Barray%7D%7Bc%7D%20x_t%20%5Cin%20R%5En%20%5Ctext%7B%3A%20input%20vectors%20at%20time%20%7Dt%20%5C%5C%20h_%7Bt-i%7D%20%5Cin%20R%5En%20%5Ctext%7B%3A%20state%20vectors%20at%20time%20%7Dt-i%20%5C%5C%20y_t%20%5Cin%20R%5Ek%20%5Ctext%7B%3A%20output%20vectors%20at%20time%20%7Dt%20%5C%5C%20W%5Ei%2C%20W%5Eh%20%5Cin%20R%5E%7Bn%20%5Ctimes%20n%7D%2C%20W%5Eo%20%5Cin%20R%5E%7Bk%20%5Ctimes%20n%7D%20%5Ctext%7B%3A%20trainable%20variables%7D%20%5C%5C%20%5Csigma_i%20%5Ctext%7B%3A%20activation%20functions%7D%20%5Cend%7Barray%7D">
</p>

## References

* [Prof. Hung-Yi Lee](http://speech.ee.ntu.edu.tw/~tlkagk/index.html)

