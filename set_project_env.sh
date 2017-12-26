#!/usr/bin/bash
#
# Sets project environment

USER=$(whoami)

whereami() {
			if [[ $(hostname) == *"greyostrich.stats.ox.ac.uk"* ]]; then
				  HOST="ostrich"
			else
				  HOST="local"
			fi
}

# Detect host
whereami


export PROJECT_ROOT=$PWD
# Add working directory to python path
export PYTHONPATH=$PWD:$PYTHONPATH
