#!/usr/bin/env bash

if [ $# -ne 3 ]; then
    echo -e "\033[1;31mERROR: $0 arguments\033[0m"
    echo -e "\033[1;31mExpected: $0 ENVIORNMENT_DIR"\
            "PYTHON_VERSION REQUIREMENTS_FILE\033[0m"
    exit 1
fi

VENV_DIR=$1
PY_VERSION=$2
REQUIREMENTS=$3

if [ ! -d $VENV_DIR ]; then 
    $PY_VERSION -m venv $VENV_DIR ||\
    (echo -e "\033[1;31mERROR: $0 could not set up environment\033[0m" \
    && exit 1)
else
    echo "Python environment already exists"
fi

if [ -f $REQUIREMENTS ]; then
    source $VENV_DIR/bin/activate
    pip install -r $REQUIREMENTS 
else
    echo -e "\033[1;33mWARNING: could not find the"\
        "requirements file $REQUIREMENTS\033[0m"
fi

echo -e "INSTRUCTIONS:\n"\
        "\t- activation: source $VENV_DIR/bin/activate\n"\
        "\t- deactivation: deactivate"
