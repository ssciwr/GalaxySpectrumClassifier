Examples
========

The notebooks below are rendered from the ``notebooks/`` directory of the
repository and are not executed during the documentation build. To run one,
start it from within ``notebooks/``, where its relative paths to ``configs/``
and ``data/`` resolve.

.. toctree::
   :maxdepth: 1

   dataset_examples
   simpletrainer_examples
   epochtrainer_examples
   inference_examples

Example configurations
----------------------

These are the complete YAML configurations used by the example notebooks.
Their relative paths assume that the calling process runs from ``notebooks/``.
Configurations containing ``__main__`` import paths also require the helper
functions or model classes defined in the corresponding notebook.

Basic TabularDataset
~~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../configs/dataset_example.yaml
   :language: yaml
   :caption: configs/dataset_example.yaml

TabularDataset with preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../configs/dataset_preprocessing_example.yaml
   :language: yaml
   :caption: configs/dataset_preprocessing_example.yaml

SimpleTrainer with a random forest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../configs/binary_classsifier_simple_example.yaml
   :language: yaml
   :caption: configs/binary_classsifier_simple_example.yaml

SimpleTrainer with a torch model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../configs/binary_classsifier_simple_example_torch.yaml
   :language: yaml
   :caption: configs/binary_classsifier_simple_example_torch.yaml

EpochTrainer with a torchvision MLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../configs/binary_classifier_epoch_example.yaml
   :language: yaml
   :caption: configs/binary_classifier_epoch_example.yaml

EpochTrainer with a custom convolutional model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../configs/binary_classifier_epoch_custom_model_example.yaml
   :language: yaml
   :caption: configs/binary_classifier_epoch_custom_model_example.yaml
