
.PHONY: tests
tests:
	pytest tests

.PHONY: get-base-data
get-base-data:
	cd data && sh ./prepare_data.sh

#.PHONY: get-tiny-data
#get-tiny-data:
#	cd data && sh get_tiny_data.sh
#
#.PHONY: clean-tiny-data
#clean-tiny-data:
#	rm -r ./data/dataset/split/tiny*