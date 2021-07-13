# Breaking down the Jupyter notebook monolith: Building reliable notebook-based pipelines with Ploomber

Author: [Eduardo Blancas](https://twitter.com/edublancas)

This poster is an interactive notebook that shows how to develop a notebook-based pipeline using [Ploomber](https://github.com/ploomber/ploomber). To start, [click here](https://mybinder.org/v2/gh/edublancas/scipy-2021/main?urlpath=lab/tree/index.ipynb) or on the button below:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edublancas/scipy-2021/main?urlpath=lab/tree/index.ipynb)

**Note:** It may take a few seconds for the notebook to load.

Prefer a video presentation? [Click here](https://www.youtube.com/watch?v=XCgX1AszVF4).

Got questions? Reach out to me via [Twitter](https://twitter.com/edublancas).

## Running it locally

If you prefer to run things locally (requires `conda`):

```sh
pip install invoke

invoke setup

conda activate scipy-2021

jupyter lab
```

Then open `index.ipynb`.