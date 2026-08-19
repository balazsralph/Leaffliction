VENV		:= .venv
PYTHON		:= $(VENV)/bin/python
PIP			:= $(VENV)/bin/pip
REQUIREMENTS	:= requirements.txt

.PHONY: all setup install package clean fclean re help

all: setup

setup: $(VENV)/.installed

$(VENV)/.installed: $(REQUIREMENTS)
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@touch $@

install: setup

# Bundle every trained plant (all model_*.keras + class_names_*.json + the
# work_directory) into one dataset.zip, then write its sha1 to signature.txt.
# Run AFTER training all plants (e.g. Train.py ./DATA/Apple/ and ./DATA/Grape/).
package:
	@if [ -z "$(wildcard model_*.keras)" ]; then \
		echo "No model_*.keras found — train a plant first, e.g."; \
		echo "  $(PYTHON) Train.py ./DATA/Apple/"; \
		exit 1; \
	fi
	rm -f dataset.zip signature.txt
	zip -r dataset.zip model_*.keras class_names_*.json work_directory
	sha1sum dataset.zip | awk '{print $$1}' > signature.txt
	@echo "signature.txt -> $$(cat signature.txt)"

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete


fclean: clean
	rm -rf $(VENV)

re: fclean setup
