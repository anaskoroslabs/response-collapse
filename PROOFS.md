# Appendix B (supplement): Proof Path for Theorem 1

(LaTeX source; compiles standalone within the paper preamble.)

```latex
\section*{Appendix B: Proof Path for Theorem 1}

\paragraph{Injection kernel.} With $k=k(T)$ bins, posterior
$\mathrm{Dir}(a_1,\dots,a_k)$, $A_T=\sum_j a_j=T+O(1)$,
$\hat\theta_j=a_j/A_T$, the predictive density is
$q_T(y)=k\hat\theta_{b(y)}$, and observing $x$ gives
$q_T^{+x}(y)=k\,(a_{b(y)}+\mathbf 1\{b(y)=b(x)\})/(A_T+1)$. The live-input
injection kernel is therefore
$G_T(x)(y)=\frac{k}{A_T+1}[\mathbf 1\{b(y)=b(x)\}-\hat\theta_{b(y)}]$, and
in differences the renormalization background cancels:
\[
G_T(x{+}\delta)-G_T(x)=\tfrac{k}{A_T+1}\big[\mathbf 1_{B_{b(x+\delta)}}-\mathbf 1_{B_{b(x)}}\big].
\]
All three rates are this one transport object read through three
elicitations, at resolution-scale perturbations $|\delta|=s h$, $h=1/k$.

\paragraph{Pointwise ($T^{\alpha-1}$).} $\ell_{\mathrm{pt}}$ reads the
kernel at $y_0$: nonzero only when the perturbation crosses $B_{b(y_0)}$,
an event of probability $\Theta(h)$ (bounded truth), with magnitude $k/T$.
So $\mathbb E|\Delta\ell|=\Theta(h\cdot k/T)=\Theta(T^{-1})$; dividing by
$|\delta|\asymp h$ gives $L_X=\Theta(k/T)=\Theta(T^{\alpha-1})$.

\paragraph{Smooth global ($T^{-1}$).} With $\bar a_j=k\int_{B_j}a$,
$\ell_a(G_T(x{+}\delta)-G_T(x))=(\bar a_{b(x+\delta)}-\bar a_{b(x)})/(A_T{+}1)$,
and $C^1$ smoothness gives $|\bar a_{b(x+\delta)}-\bar a_{b(x)}|=\Theta(h)$
on positive measure. So $\mathbb E|\Delta\ell_a|=\Theta(h/T)$; dividing by
$h$: $L_X=\Theta(T^{-1})$, independent of $k$ to first order.

\paragraph{Full predictive ($T^{3\alpha/2-1}$).} The $L^2$ distance between
distinct bin indicators is $\|\mathbf 1_{B_i}-\mathbf
1_{B_j}\|_2=\sqrt{2h}$, so on a bin change
$\|G_T(x{+}\delta)-G_T(x)\|_2=\Theta(k\sqrt h/T)=\Theta(\sqrt k/T)$; the
change probability is bounded below over the compact $s$-range. Dividing by
$h$: $L_X=\Theta(k^{3/2}/T)=\Theta(T^{3\alpha/2-1})$; at $\alpha=\tfrac12$,
$T^{-1/4}$. The rate is a \emph{transport} rate: amplitude $O(k/T)$ times
the $L^2$ size $O(\sqrt h)$ of a moved bin, per resolution-scale
perturbation $h$.

\paragraph{Perturbation-scale plateau.} For $s\in[s_{\min},s_{\max}]$ the
crossing probability scales linearly in $s$ and the quotient divides by
$s h$, so fitted exponents are $s$-independent over the admissible range;
measured spread $<0.01$ across $s\in\{0.5,1,2,4\}$ (Robustness paragraph).


```
