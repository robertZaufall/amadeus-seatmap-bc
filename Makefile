.PHONY: llms

LLMS_INCLUDE   = *.py,.github/workflows/*,.env.template
LLMS_EXCLUDE   = docs/*,data/*,llms.txt,/__pycache__,.DS_Store

# Default target
all: llms

# llms
llms:
	pip install gitingest
	gitingest . -o llms.txt -i "$(LLMS_INCLUDE)" -e "$(LLMS_EXCLUDE)"

# Help
help:
	@echo "Available targets:"
	@echo "  all            - Run llms"
	@echo "  llms           - Generate llms documentation"
	@echo "  help           - Show this help message"
