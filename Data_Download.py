import kagglehub

destination = "./data"

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia", path=destination)

print("Path to dataset files:", path)