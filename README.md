### Intro

This is a fork of https://github.com/climatechange-ai-tutorials/downscaling-climate-projections. The original repo housed the entire code in a single notebook. This repo refactors that code into a codebase with type hints, doc strings, and logical modules.

### Getting Started

To run the main script,

```
poetry install
cd src
poetry run python3 main.py
```

### Answered Questions

> Look at the coarse ("GCM-like") panel and the bars. Why do the two locations show a similar change, even though the high-resolution simulation says they change differently? What information has been lost?

This can be answered in general. If we have some $f(\mathbf{x}) $

> Compare the bias_mean value in the table with bias_P98 and bias_RX1day. Which aspects of precipitation does the model reproduce well, and which does it struggle with? Why might under-estimating extremes be dangerous when the goal is to plan for floods?

> Two different models could have similar RMSE but very different extreme biases. Why, then, is it important to report a battery of diagnostics rather than one number?

The answer is in the question. Qualitatively different models can have the same macroscopic or average behavior. This is relevant if one cares about forecasts at a particular subregion.  More concretely, suppose two models $A$ and $B$ that forecast rainfall on sites $S=\{s_1, s_2, \cdots, s_n\}$ and $n$ even. Moreover, $A$ forecasts perfectly on $\S_1=\{s_1, \cdots, s_{n/2}\}$ and $B$ forecasts perfectly on $S_2=\{s_{n/2+1}, \cdots, s_{n}\}$. Where $A$ And $B$ do not forecast perfectly, they produce random forecasts with the same mean error $\epsilon$. $A$ and $B$ have the same average error on $S$ but are only accurate on the disjoint sets $S_1$ and $S_2$ respectively. Now imagine that a heavy storm will impact the $s_1$ and it's imperative to forecast rainfall in the region, since if there's too much rainfall there will need to be evacuations for likely flooding. By construction, we know that model $A$ predicts perfectly on $\{s_1\} \subset S_1$, but an analysis of models $A$ and $B$ using average error on $S$ alone provides no such direction.

### Original Repo Author
González-Abad, J. (2026). Statistical Downscaling of Climate Projections with Deep Learning [Tutorial]. In Climate Change AI Summer School. Climate Change AI. https://doi.org/10.5281/zenodo.21446887

```
@misc{gonzalez2026downscaling,
  title={Statistical Downscaling of Climate Projections with Deep Learning},
  author={González-Abad, Jose},
  year={2026},
  organization={Climate Change AI},
  type={Tutorial},
  doi={https://doi.org/10.5281/zenodo.21446887},
  booktitle={Climate Change AI Summer School},
  howpublished={\url{https://github.com/climatechange-ai-tutorials/downscaling-climate-projections}}
}
```
