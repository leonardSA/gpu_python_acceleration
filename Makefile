SCRIPTS_DIR=scripts
MAKE_GRAPHS_SCRIPT=make_graphs.sh

PY_REQ=.py_requirements.txt
PIP_VER=pip3
PY_VER=python3.8
PY_ENV_DIR=.environment

SRC_DIR=src
MATRIX_NAIVE_SRC=matrix_naive.py

OUT_DIR=output

environment-setup:
	./$(SCRIPTS_DIR)/configure_python_env.sh $(PY_VER) $(PIP_VER) $(PY_ENV_DIR) $(PY_REQ)

graphs: matrix-naive-graphs

matrix-naive-graphs: output
	make matrix-naive-graphs -C $(SRC_DIR)
	make matrix-graphs -C $(SCRIPTS_DIR)
	cd $(OUT_DIR) ; ./$(MAKE_GRAPHS_SCRIPT) $(PY_VER) $(MATRIX_NAIVE_SRC)


output:
	if [ ! -d $(OUT_DIR) ]; then mkdir $(OUT_DIR); fi

clean:
	rm -rf $(OUT_DIR)
