###############################################################################
# Common make values.
lib    := par_treasure_hunt
run    := uv run
python := $(run) python
pyright := $(run) pyright
ruff  := $(run) ruff
twine  := $(run) twine
build  := uv build


#export UV_LINK_MODE=copy
export PIPENV_VERBOSITY=-1
##############################################################################
# Run the app.
.PHONY: run
run:		# Run app
	$(run) $(lib)

.PHONY: app_help
app_help:		# Show app help
	$(run) $(lib)  --help

##############################################################################
.PHONY: uv-lock
uv-lock:
	uv lock

.PHONY: uv-sync
uv-sync:
	uv sync

.PHONY: setup
setup: uv-lock uv-sync	        # use this for first time run

.PHONY: resetup
resetup: remove-venv setup			# Recreate the virtual environment from scratch

.PHONY: remove-venv
remove-venv:			# Remove the virtual environment
	rm -rf .venv

.PHONY: depsupdate
depsupdate:			# Update all dependencies
	uv sync -U

.PHONY: depsshow
depsshow:			# Show the dependency graph
	uv tree

.PHONY: shell
shell:			# Start shell inside of .venv
	$(run) bash
##############################################################################
# Checking/testing/linting/etc.

.PHONY: format
format:                         # Reformat the code with ruff.
	$(ruff) format src/$(lib)

.PHONY: lint
lint:                           # Run ruff lint over the library
	$(ruff) check src/$(lib) --fix

.PHONY: lint-unsafe
lint-unsafe:                           # Run ruff lint over the library
	$(ruff) check src/$(lib) --fix --unsafe-fixes

.PHONY: typecheck
typecheck:			# Perform static type checks with pyright
	$(pyright)

.PHONY: typecheck-stats
typecheck-stats:			# Perform static type checks with pyright and print stats
	$(pyright) --stats

.PHONY: checkall
checkall: format lint typecheck 	        # Check all the things

.PHONY: pre-commit	        # run pre-commit checks on all files
pre-commit:
	pre-commit run --all-files

.PHONY: pre-commit-update	        # run pre-commit and update hooks
pre-commit-update:
	pre-commit autoupdate

##############################################################################
# Package/publish.
.PHONY: package
package: clean			# Package the library
	$(build)

.PHONY: spackage
spackage:			# Create a source package for the library
	$(build) --sdist

.PHONY: test-publish
test-publish: package		# Upload to testpypi
	$(publish) upload --index testpypi --check-url

.PHONY: publish
publish: package		# Upload to pypi
	$(publish) upload --check-url

##############################################################################
# Utility.

.PHONY: profile
profile:
	$(run) scalene -m $(lib)

.PHONY: profile2
profile2:			# Profile app with pyinstrument and generate report
	$(run) pyinstrument -r json -o profile_report.json -m $(lib)

.PHONY: get-venv-name
get-venv-name:		# get venv python location
	$(run) which python

.PHONY: repl
repl:				# Start a Python REPL
	$(python)

.PHONY: clean
clean:				# Clean the build directories
	rm -rf build dist $(lib).egg-info

.PHONY: help
help:				# Display this help
	@grep -Eh "^[a-z]+:.+# " $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.+# "}; {printf "%-20s %s\n", $$1, $$2}'

##############################################################################
# Housekeeping tasks.
.PHONY: housekeeping
housekeeping:			# Perform some git housekeeping
	git fsck
	git gc --aggressive
	git remote update --prune
