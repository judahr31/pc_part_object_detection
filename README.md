# AI Camp Computer Vision YOLO Project Code


This folder includes jupyter notebooks for the Computer Vision YOLO Project training.

## Directory Structure

### README.md: This file

### yolov4_cfg folder 
This is a folder for keeping yolov4 configuration templates
- yolov4-custom.cfg - template yolov4 configuration file to use for fast training

### gather_data folder
This is a folder for code to gather data via Azure Bing API
- bing_image_search_api.ipynb - code to access bing API to gather images
- bing_image_search_api.md

### prep_data folder
This is a folder for code to prepare data for yolo training 
- prep_lblbx_labels.ipynb - Student project notebook to complete
- prep_lblbx_labels.md
- prep_lblbx_labels_solution.ipynb - Example solution for prep_lblbx_labels.ipynb 
- prep_lblbx_labels_solution.md
- json_test.csv - sample csv with json snippet to test write_label function
- labelboxfile.csv - sample csv to help test code

### split_data folder
This is a folder for code to split the prepared data into three sets (i.e. train,valid, and test)
- split_data.ipynb - Student project notebook to complete
- split_data.md
- split_data_solution.ipynb - Example solution for split_data.ipynb 
- split_data_solution.md
- split_data_alternative_solution.ipynb - Alternative solution to split the data
- split_data_alternative_solution.md

### evaluate_model folder
This is a folder for code to evaluate a trained model
- ai.py - helper functions to feed forward and make predictions
- evaluate_model.ipynb - Student project notebook to complete
- evaluate_model.md
- evaluate_model_solutions.ipynb - Example solution for evaluate_model.ipynb 
- evaluate_model_solutions.md
