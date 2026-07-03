# InvertibleCE / ICE — Attribution

The Python files in this directory are adapted with minimal changes from the public
implementation accompanying:

> **Invertible Concept-based Explanations for CNN Models with Non-negative Concept Activation Vectors**
> Ruihan Zhang, Prashan Madumal, Tim Miller, Krista A. Ehinger, Benjamin I. P. Rubinstein.
> *Proceedings of the AAAI Conference on Artificial Intelligence, 2021.*

Original repository: <https://github.com/zhangrh93/InvertibleCE>

We use ICE as one of the comparison baselines reported in our paper (see
`src/lad/decomposition.py` and `src/lad/metrics.py` for the LAD-side wiring). The
original author attribution is preserved in every file's module docstring. No algorithmic
modifications have been made; the only changes are import-path housekeeping required by
Python packaging.

If you use this code, please cite the ICE paper in addition to LAD.
