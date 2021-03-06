
.PHONY: run train_ir

DATA_HOME=data
CACHE_HOME=cache

.PRECIOUS: $(CACHE_HOME)/f/$(MODEL_F).pth

venv:
	python3 -m virtualenv venv

$(CACHE_HOME):
	@mkdir -p $@

$(DATA_HOME):
	@mkdir -p $@

$(DATA_HOME)/cifar-10: $(DATA_HOME) src/get_cifar10.py
	@python src/get_cifar10.py --path_data $(DATA_HOME) --path_cifar10 $@

train_ir: $(DATA_HOME)/cifar-10 src/train_ir.py
	@python src/train_ir.py --path_cifar10 $(DATA_HOME)/cifar-10 --path_saved_model $(CACHE_HOME)/model/saved_model.pth

run: train_ir
	$(info done!)

