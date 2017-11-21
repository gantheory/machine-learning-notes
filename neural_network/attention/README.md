# Attention

The main idea of attention mechanism is to know what the decoder state should be at time t in a sequence-to-sequence model. Besides, attention-based model are classified into two categories, global and local.

Common to these two types of attention, both of them take the hidden state ![h_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24h_t%24%24) from the top of a stacking RNN cells (LSTM, GRU, ...) layer, and want to derive a context vector ![c_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24c_t%24%24) such that
<!---
\begin{align*}
\widetilde{h_t} &= \tanh(W^c [h_t^T, c_t^T]^T) \\
y_t &= softmax(W^o \widetilde{h_t})
\end{align*}
-->
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cbegin%7Balign*%7D%20%5Cwidetilde%7Bh_t%7D%20%26%3D%20%5Ctanh%28W%5Ec%20%5Bh_t%5ET%2C%20c_t%5ET%5D%5ET%29%20%5C%5C%20y_t%20%26%3D%20softmax%28W%5Eo%20%5Cwidetilde%7Bh_t%7D%29%20%5Cend%7Balign*%7D">
</p>
However, global and local attention differ in how the context vector ![c_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24c_t%24%24) is derived.

## Global

In global attention, we want to compute a variable-length alignment vector  ![\alpha_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%5Calpha_t%24%24) ,whose length equals the number of time steps on the source side (encoder), based on the encoder hidden states ![h_{s_i}](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24h_%7Bs_i%7D%24%24).
<!---
\begin{align*}
\alpha_t(i) &= \text{align}(h_t, h_{s_i}) \\
&= \frac{\text{exp}(\text{score}(h_t, h_{s_i}))))}{\sum_{s'} \text{exp}(\text{score}(h_t, h_{s'})))}
\end{align*}
-->
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cbegin%7Balign*%7D%20%5Calpha_t%28i%29%20%26%3D%20%5Ctext%7Balign%7D%28h_t%2C%20h_%7Bs_i%7D%29%20%5C%5C%20%26%3D%20%5Cfrac%7B%5Ctext%7Bexp%7D%28%5Ctext%7Bscore%7D%28h_t%2C%20h_%7Bs_i%7D%29%29%29%29%7D%7B%5Csum_%7Bs%27%7D%20%5Ctext%7Bexp%7D%28%5Ctext%7Bscore%7D%28h_t%2C%20h_%7Bs%27%7D%29%29%29%7D%20%5Cend%7Balign*%7D">
</p>
such that
<!---
\begin{align*}
c_t = \sum_i \alpha_{t(i)} h_{s_i}
\end{align*}
-->
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cbegin%7Balign*%7D%20c_t%20%3D%20%5Csum_i%20%5Calpha_%7Bt%28i%29%7D%20h_%7Bs_i%7D%20%5Cend%7Balign*%7D">
</p>
where score could be any reasonable function such as cosine similarity, neural network, etc. The following are score functions used by Luong and Bahdanau.
<!---
$$
\text{score}(h_t, h_{s_i}) =
\begin{cases}
h_t^T W h_{s_i},& \text{Luong's multiplicative style} \\
\upsilon^T \tanh(W_1 h_t + W_2 h_{s_i}),& \text{Bahdanau's additive style}
\end{cases}
$$
-->
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%20%5Ctext%7Bscore%7D%28h_t%2C%20h_%7Bs_i%7D%29%20%3D%20%5Cbegin%7Bcases%7D%20h_t%5ET%20W%20h_%7Bs_i%7D%2C%26%20%5Ctext%7BLuong%27s%20multiplicative%20style%7D%20%5C%5C%20%5Cupsilon%5ET%20%5Ctanh%28W_1%20h_t%20&plus;%20W_2%20h_%7Bs_i%7D%29%2C%26%20%5Ctext%7BBahdanau%27s%20additive%20style%7D%20%5Cend%7Bcases%7D%20%24%24">
</p>

## Local

The main drawback of global attention is the expensive computation and potential impractical to translate longer sequences, e.g., paragraphs or documents. Therefore, we can compute ![\alpha_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%5Calpha_t%24%24) with limited hidden states of the encoder.

In details, we can compute an aligned position (a scalar) ![p_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24p_t%24%24) such that ![\alpha_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%5Calpha_t%24%24) and ![c_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24c_t%24%24)  are derived from the set of hidden states within the window ![range](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%5Bp_t%20-%20D%2C%20p_t%20&plus;D%5D%24%24); ![D](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24D%24) is empirically selected. Note that in local approach, our ![\alpha_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%5Calpha_t%24%24) is (2D+1)-dimensional.

In [Luong et al. 2015](https://arxiv.org/abs/1508.04025), they proposed two ways to determine ![p_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24p_t%24%24).

<!---
$$
p_t =
\begin{cases}
t,& \text{Monotomic alignment} \\
S \cdot \text{sigmoid}(\upsilon_p^T \tanh(W_p h_t)), & \text{Predictive alignment}
\end{cases}
$$
-->
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%20p_t%20%3D%20%5Cbegin%7Bcases%7D%20t%2C%26%20%5Ctext%7BMonotomic%20alignment%7D%20%5C%5C%20S%20%5Ccdot%20%5Ctext%7Bsigmoid%7D%28%5Cupsilon_p%5ET%20%5Ctanh%28W_p%20h_t%29%29%2C%20%26%20%5Ctext%7BPredictive%20alignment%7D%20%5Cend%7Bcases%7D%20%24%24">
</p>
where  S is the source sentence length to let ![p_t \in 0 ~ S](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%20p_t%20%5Cin%20%5B0%2C%20S%5D%20%24%24) as a result of sigmoid.

Besides, we can further favor alignment point near ![p_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24p_t%24%24) by a Gaussian distribution centered around ![p_t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24p_t%24%24).
<!---
$$
\alpha_t(i) = \text{align}(h_t, h_{s_i}) \text{exp}(- \frac{(i - p_t)^2}{2 \sigma^2})
$$
-->
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%20%5Calpha_t%28i%29%20%3D%20%5Ctext%7Balign%7D%28h_t%2C%20h_%7Bs_i%7D%29%20%5Ctext%7Bexp%7D%28-%20%5Cfrac%7B%28i%20-%20p_t%29%5E2%7D%7B2%20%5Csigma%5E2%7D%29%20%24%24">
</p>
Empirically ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%24%24%5Csigma%20%3D%20%5Cfrac%7BD%7D%7B2%7D%24%24)

## Implementation Details

In paper works of [Luong et al. 2015](https://arxiv.org/abs/1508.04025), and [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf), they use hidden states at the top LSTM layers in both the encoder and decoder. 

While, in the implementation of [Google NMT Model](https://github.com/tensorflow/nmt/blob/master/nmt/gnmt_model.py#L167), they use the hidden states at the bottom layer of the decoder.

## References

* [Luong et al. 2015](https://arxiv.org/abs/1508.04025)
* [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf)
* [Google NMT](https://github.com/tensorflow/nmt)
