SHELL := /bin/bash
DATA_DIR := data
DATASET_DIR := $(DATA_DIR)/dataset
FLAT_DIR := $(DATASET_DIR)/flat
FLAT_ANNO_DIR := $(DATASET_DIR)/flat_anno
FLAT_ANNO_COCO_DIR := $(DATASET_DIR)/flat_anno_coco
PREPROCESS_SCRIPTS := exdark/data/preprocess
DOWNLOAD_URL := https://drive.usercontent.google.com/download?id=1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx&export=download&authuser=0&confirm=t&uuid=3232338f-26c7-4599-9f35-f5c93b8241e4&at=AO7h07dJS6_aNoNQUhl1VrZ_c3gt:1726826664489
DOWNLOAD_HVICID_URL := https://drive.usercontent.google.com/download?id=1JIVDlosrrPBXDn8rh_Xdi-Dzmk_VhlgU&export=download&confirm=t&uuid=cfcc0052-1517-4367-8d7d-898926888ef0&at=AIrpjvNn2klM59SEoBvwi5zL361U%3A1736412257826

.PHONY: data-setup data-download data-process data-split data-cleanup data-full-cleanup data-hvicid tests

# Main target that runs the full data preparation pipeline
data-setup: data-download data-process data-split data-temp-cleanup data-hvicid

# Download and extract data
data-download:
	wget '$(DOWNLOAD_URL)' -O $(DATA_DIR)/ExDark.zip && \
	cd $(DATA_DIR) && \
	unzip ExDark.zip && \
	unzip ExDark_Annno.zip && \
	rm -rf ExDark.zip __MACOSX


# Process the data files
data-process:
	mkdir -p $(DATASET_DIR) && \
	python $(PREPROCESS_SCRIPTS)/parse_imgclasslist_classes_idx.py $(DATA_DIR)/imageclasslist.txt $(DATASET_DIR)/imageclasslist.txt.coco && \
	python $(PREPROCESS_SCRIPTS)/names_tolower.py $(DATA_DIR)/ExDark && \
	python $(PREPROCESS_SCRIPTS)/names_tolower.py $(DATA_DIR)/ExDark_Annno && \
	mkdir -p $(FLAT_DIR) && \
	python $(PREPROCESS_SCRIPTS)/move_imgs_to_flat_dir.py $(DATA_DIR)/ExDark $(FLAT_DIR) && \
	mkdir -p $(FLAT_ANNO_DIR) && \
	python $(PREPROCESS_SCRIPTS)/move_imgs_to_flat_dir.py $(DATA_DIR)/ExDark_Annno $(FLAT_ANNO_DIR) && \
	rm -rf $(DATA_DIR)/ExDark $(DATA_DIR)/ExDark_Annno && \
	mkdir -p $(FLAT_ANNO_COCO_DIR) && \
	python $(PREPROCESS_SCRIPTS)/parse_anno_files_classes.py $(FLAT_ANNO_DIR) $(FLAT_ANNO_COCO_DIR)


# Split data into train/test/val sets
data-split:
	python $(PREPROCESS_SCRIPTS)/train_test_val.py $(DATASET_DIR)/imageclasslist.txt.coco $(DATASET_DIR)/split $(FLAT_DIR) $(FLAT_ANNO_COCO_DIR) ,

data-hvicid:
	mkdir -p $(DATASET_DIR)/hvicid && \
	wget '$(DOWNLOAD_HVICID_URL)' -O $(DATASET_DIR)/hvicid/HVICID.zip && \
	cd $(DATASET_DIR)/hvicid && \
	unzip HVICID.zip && \
	rm HVICID.zip && \
	rm -rf __MACOSX

# Cleanup temporary directories
data-temp-cleanup:
	rm -rf $(FLAT_DIR) $(FLAT_ANNO_DIR) $(FLAT_ANNO_COCO_DIR) $(DATASET_DIR)/imageclasslist.txt.coco

data-full-cleanup:
	rm -rf $(DATA_DIR)

tests:
	pytest tests

lint:
	ruff check exdark tests

format:
	ruff format exdark tests