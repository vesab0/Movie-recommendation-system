#!/usr/bin/env python3
import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

print("Path to dataset files:", path)
