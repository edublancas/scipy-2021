from invoke import task

import jupytext


@task
def setup(c):
    c.run('conda env create --file environment.dev.yml --force')
    c.run('conda env export --no-build --file environment.yml'
          ' --name scipy-2021')


@task
def convert(c):
    nb = jupytext.read('index.md')
    jupytext.write(nb, 'index.ipynb')