#!/usr/bin/env bash

if [ $# -ne 4 ]; then
    echo -e "\033[1;31mERROR: $0 arguments\033[0m"
    echo -e "\033[1;31mExpected: $0 ENVIORNMENT_DIR"\
            "PYTHON_VERSION REQUIREMENTS_FILE\033[0m"
    exit 1
fi

PY_VERSION=$1
PIP_VERSION=$2
VENV_DIR=$3
REQUIREMENTS=$4

if [ ! -d $VENV_DIR ]; then 
    $PY_VERSION -m venv $VENV_DIR ||\
    (echo -e "\033[1;31mERROR: $0 could not set up environment\033[0m" \
    && exit 1)
else
    echo "Python environment already exists"
fi

if [ -f $REQUIREMENTS ]; then
    source $VENV_DIR/bin/activate
    $PIP_VERSION install -r $REQUIREMENTS 
else
    echo -e "\033[1;33mWARNING: could not find the"\
        "requirements file $REQUIREMENTS\033[0m"
fi

echo -e "INSTRUCTIONS:\n"\
        "\t- activation: source $VENV_DIR/bin/activate\n"\
        "\t- deactivation: deactivate"
