SCRIPTS_DIR=scripts
PY_REQ=.py_requirements.txt
PIP_VER=pip3
PY_VER=python3.8
PY_ENV_DIR=.environment

environment-setup:
	./$(SCRIPTS_DIR)/configure_python_env.sh $(PY_VER) $(PIP_VER) $(PY_ENV_DIR) $(PY_REQ)
