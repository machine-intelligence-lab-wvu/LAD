# CRAFT — Attribution

The Python files in this directory are adapted with minimal changes from the public
implementation accompanying:

> **CRAFT: Concept Recursive Activation FacTorization for Explainability**
> Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux,
> Julien Colin, Rémi Cadène, Thomas Serre.
> *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.*

We use the Sobol' total-order estimators (Jansen, Homma, Janon, Glen, Saltelli) and the
quasi-random / replicated-design samplers (Sobol, Halton, Latin Hypercube) to compute
concept-importance indices in `lad.metrics`. The original author attribution is preserved
in every file's module docstring. No algorithmic modifications have been made; the only
changes are import-path housekeeping required by Python packaging.

If you use this code, please cite the CRAFT paper in addition to LAD.
