# Python Virtual Environments

A virtual environment is an isolated folder that holds its own Python interpreter and its own set of installed packages. Each project can have its own environment, so a package upgrade in one project cannot break another project.

## Creating and using one

Python ships with the `venv` module. To create an environment, run `python -m venv .venv` inside your project folder. To activate it on Linux or macOS, run `source .venv/bin/activate`. On Windows, run `.venv\Scripts\activate`. Once active, the command `pip install requests` installs the package only inside that environment. Run `deactivate` to leave the environment.

## Why they matter

Different projects often need different versions of the same library. Without virtual environments, every project would share one global set of packages, and version conflicts would be common. Virtual environments also make a project easier to reproduce on another computer.

## Sharing dependencies

Developers record the exact packages a project needs in a file called `requirements.txt`. The command `pip freeze > requirements.txt` writes the installed packages and their versions to that file. Another person can then run `pip install -r requirements.txt` to rebuild the same environment. The `.venv` folder itself should not be committed to Git, so it is normally listed in `.gitignore`.
