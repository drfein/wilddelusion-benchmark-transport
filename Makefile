.PHONY: install list doctor verify plan reproduce

install:
	python -m pip install -e '.[dev]'

list:
	wd-reproduce list

doctor:
	wd-reproduce doctor

verify:
	wd-reproduce verify

plan:
	wd-reproduce run all --dry-run --no-resume --allow-api --allow-gpu

reproduce:
	wd-reproduce run all --allow-api --allow-gpu
