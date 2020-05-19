SCRIPTS_DIR=scripts
PY_REQ=.py_requirements.txt
PY_VER=python3.8
PY_ENV_DIR=.environment

environment-setup:
	./$(SCRIPTS_DIR)/configure_python_env.sh $(PY_ENV_DIR) $(PY_VER) $(PY_REQ)


