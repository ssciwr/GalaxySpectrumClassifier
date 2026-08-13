Examples
========

The notebooks below are rendered from the ``notebooks/`` directory of the
repository and are not executed during the documentation build. To run one,
start it from within ``notebooks/``, where its relative paths to ``configs/``
and ``data/`` resolve.

.. toctree::
   :maxdepth: 1

   simpletrainer_examples

.. Adding a notebook takes two steps: write doc/<name>.nblink containing
   {"path": "../notebooks/<name>.ipynb"}, then list <name> in the toctree
   above. Still to come:

     epochtrainer_examples
     conv1d_model
     regression
