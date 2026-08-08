VENV		:= .venv
PYTHON		:= $(VENV)/bin/python
PIP			:= $(VENV)/bin/pip
REQUIREMENTS	:= requirements.txt

.PHONY: all setup install clean fclean re help

all: setup

setup: $(VENV)/.installed

$(VENV)/.installed: $(REQUIREMENTS)
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@touch $@

install: setup

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete


fclean: clean
	rm -rf $(VENV)

re: fclean setup
