SCRIPTS_DIR=scripts
MAKE_GRAPHS_SCRIPT=make_graphs.sh

PY_REQ=.py_requirements.txt
PIP_VER=pip3
PY_VER=python3.8
PY_ENV_DIR=.environment

SRC_DIR=src
MATRIX_NAIVE_SRC=main.py

DOC_DIR=doc
OUT_DIR=output

environment-setup:
	./$(SCRIPTS_DIR)/configure_python_env.sh $(PY_VER) $(PIP_VER) $(PY_ENV_DIR) $(PY_REQ)

graphs: output
	make matrix-graphs -C $(SRC_DIR)
	make matrix-graphs -C $(SCRIPTS_DIR)
	cd $(OUT_DIR) ; ./$(MAKE_GRAPHS_SCRIPT) $(PY_VER) $(MATRIX_NAIVE_SRC) 30 10 1500
	cd $(OUT_DIR) ; ./$(MAKE_GRAPHS_SCRIPT) $(PY_VER) $(MATRIX_NAIVE_SRC) 30 10 1500 naive

.PHONY: doc
doc:
	make -C $(DOC_DIR)

output:
	if [ ! -d $(OUT_DIR) ]; then mkdir $(OUT_DIR); fi

clean:
	make clean -C $(DOC_DIR)
	rm -rf $(OUT_DIR)
