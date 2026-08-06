# Code Review: `dataset-fixes`

| ID | Severity | Code location | Summary |
|---|---|---|---|
| [H2](#h2) | High | [`TabularDataset._map_index`](src/GalaxySpectrumClassifier/data.py#L650) | Labels are selected by cached column positions rather than the configured names and order, so targets can be silently swapped. |
| [M1](#m1) | Medium | [`TabularDataset.__init__`](src/GalaxySpectrumClassifier/data.py#L422), [`TabularDataset.__getitem__`](src/GalaxySpectrumClassifier/data.py#L685) | Missing or unknown label columns are not validated and produce an `IndexError` or an incomplete target instead of the documented `ValueError`. |
| [M2](#m2) | Medium | [`TabularDataset._normalize_index`](src/GalaxySpectrumClassifier/data.py#L622) | Slice handling does not follow Python bounds semantics, and empty slices fail inside `torch.stack`. |
| [M3](#m3) | Medium | [`transform` documentation](src/GalaxySpectrumClassifier/data.py#L333), [`transform` call](src/GalaxySpectrumClassifier/data.py#L715) | The documented transform input and contract do not match the implementation. |
| [M4](#m4) | Medium | [`simpletrainer_examples.ipynb`](notebooks/simpletrainer_examples.ipynb#L71) | The example notebook still reads the pre-refactor configuration shape and fails with the updated YAML files. |

## High

<a id="h2"></a>
### H2: Label order and identity are lost during positional caching

On the first untransformed retrieval, `_map_index()` walks the frame's columns
and stores positions for names found in `label_columns`. This has two correctness
problems:

1. The resulting order follows the frame, not the documented ordered
   `label_columns` collection. With columns `a, source, extra` and
   `label_columns=["extra", "source"]`, the returned target is
   `[source, extra]`.
2. `label_indices` is cached once for the whole dataset. If a later file has the
   same named columns in a different order, those positions select different
   columns. A reproduction with the second file ordered `a, extra, source`
   returned `[extra, source]` for a dataset configured with
   `["source", "extra"]`.

This silently changes target meaning while returning tensors of the expected
shape. Resolve labels by name for each frame, preserve the configured label
order, and derive feature positions independently from each frame's columns.

## Medium

<a id="m1"></a>
### M1: Missing label configuration is not validated

`label_columns=None` is normalized to `[None]`. An unknown label name similarly
produces no matching positions. Scalar retrieval then evaluates
`self.label_indices[0]` and raises the unrelated `IndexError: list index out of
range`.

For a multi-label configuration where only some names exist, retrieval is more
dangerous: it returns a shorter target tensor without reporting the missing
column. Both outcomes contradict `__getitem__`'s documented `ValueError` for
unconfigured or unavailable targets, and the branch deleted the previous tests
for both cases.

Validate that labels are configured and that every requested label exists
before constructing feature and target tensors.

<a id="m2"></a>
### M2: Supported slices do not have normal slice semantics

`_normalize_index()` constructs a raw `range(start, stop, step)` instead of
normalizing through `slice.indices(len(self))`. Consequently:

- `dataset[-1:]` produces `-1` and raises `IndexError` instead of selecting the
  final row.
- `dataset[:len(dataset) + 1]` includes an out-of-range position instead of
  clamping to the dataset length.
- `dataset[0:0]` reaches `torch.stack([])` and raises
  `RuntimeError: stack expects a non-empty TensorList`.

Since slices are explicitly part of `DatasetProtocol` and `TabularDataset`'s
public index type, their bounds should be normalized consistently and the empty
result should have a deliberate contract.

<a id="m3"></a>
### M3: The transform docstring describes the wrong API

The constructor says `transform` receives a one-row `pd.DataFrame`, is applied
before label splitting, and must retain label columns. The implementation
instead passes `df.iloc[local_idx]`, which is a `pd.Series`, and immediately
unpacks the callable's result as `(X, y)`. No subsequent label splitting or
label-column validation occurs.

A transform written to the documented contract will therefore fail or return
invalid values. Document the actual `Series -> tuple[Tensor, Tensor]` contract,
or restore the documented DataFrame-and-split behavior.

<a id="m4"></a>
### M4: The example notebook is incompatible with the updated configs

The YAML files moved `engine`, `comment`, `na_values`, and `sep` under
`dataset.read_kwargs`. The notebook's preprocessing cell still reads
`config["dataset"]["engine"]` and `config["dataset"]["comment"]`, which no
longer exist. It also passes `sep` explicitly while expanding `read_kwargs`,
which now already contains `sep`.

With an input `.dat` file present, this cell raises `KeyError`; retaining the
old top-level keys would instead leave duplicate keyword arguments. Update the
cell to construct a dedicated set of raw-file read options rather than
re-expanding the processed-dataset CSV options.

## Verification

- Compared `dataset-fixes` with `main` from merge base
  `1db9be8a063364d21284390a4f36576b28957e89`.
- Ran the complete test suite: **147 passed, 1 warning**.
- Confirmed H2, M1, and M2 with focused temporary-file reproductions.
- `git diff --check` passed.
- Ruff/pre-commit could not be run from the existing environment because those
  executables are not installed; `uv run` could not initialize its cache under
  the active filesystem sandbox.
