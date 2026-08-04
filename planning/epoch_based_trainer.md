# Plan for epoch based trainer

## High level list of features
- adhere to SOLID principles
- no data preprocessing inside trainer, it gets a dataset from the outside and just uses it
- adheres to `TrainerProtocol`
- functions for train, validation, test
- callbacks at start_epoch, end_epoch, after_train_batch, after_val_batch, before_train, before_test, after_train, after_test
- early stopping is first class citizen and is evaluated on a given set of validation metrics
- early stopping has patience parameter (evaluation + countdown )
- early stopping triggers if patience runs out without improvement of designated metrics
- early stopping patience is reset upon found improvement of designated metrics.
- has snapshot serialization system based on torch

## Architecture and requirements, logic and API
- complexity needs to be kept minimal, do not build complicated routing code, workarounds, subclasses or helper functions where they are not absolutely necessary.
    - no complicated routing around problems with data or logic flow. If such a thing becomes necessary it's a code/architecture smell and needs to be surfaced and discussed, not worked around
- single clean hot path:
    - keep as clean and straight as possible
    - as little branching within hot path as possible, put all decisions into the constructor of the trainer whereever it's possible.
    - training:
        - is passed a train dataset and validation dataset from the outside, trainer doesn't modify them
        - trigger before_train callback if exists
        - run training on epoch
            - run training on train batch
                - trigger after_train_batch callback if exists
                - run until epoch complete
        - run validation on epoch
            - run validation on val batch
            - trigger after_val_batch if exists
            - record validation metrics
            - check early stopping
        - run after_train callback if exists when done
    - testing:
        - is passed a test dataset from the outside, trainer doesn´t modify it
        - triggers before_test if exists
        - runs test batch until test set complete
        - record testing metrics
        - runs after_test_batch if exists
        - runs after_test if exists upon completion
- allow for export to .pt and onnx in save_model
    - export format derived from config or constructor argument, respectively
- callbacks are given by dotted name, args kwargs, are instantiated with load_type where they are needed, which results in a Callable instance. This assures round-trip capability with save/load_snapshot.
- make use of type, args, kwargs passing together with established `load_type` paradigm
- configuration structure (which also gives structure of constructo args/kwargs):
    - model: model type, args, kwargs, depending on type
    - training: batch size, patience, validation metrics, used for early stopping (can be different from total metrics), training and validation callbacks
    - testing: testing callbacks, testing metrics to be recorded.
    - optimizer: type, args, kwargs
    - learning_rate_scheduler: type, args, kwargs

---
## Restrictions
- complexity needs to be kept minimal, do not build complicated routing code, workarounds, subclasses or helper functions where they are not absolutely necessary.

- no changes to dataset.py
- no changes to utils.py
- no changes to trainer.py

if any of these latter 3 restrictions cannot be satisfied without violating the primary complexity restriction, ask first and wait for review.
