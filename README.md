# Dev-requirements

## pyenv + virtualenv
```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo  'eval "$(pyenv init -)"' >> ~/.bashrc
echo  'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
```

## Poetry

#### Install poetry
```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
export PATH="$HOME/.poetry/bin:$PATH"
poetry install  # install the defined dependencies for the project
```

#### Run the script
```sh
# spawn a shell within your virtual environment
poetry shell
python <script>

# or you can directly run
poetry run python <script>
```

## IDE plugins:
    * flake8
    * black
    * mypy

## Pre-commit hook
```sh
pre-commit install
```
