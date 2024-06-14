import json
import os

def get_config_path(relative_path):
    # Directory of the script or module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

# Example of how to use the function to get the path to the configs directory
config_dir_path = get_config_path("../configs")
print("Config Directory Path:", config_dir_path)

if not os.path.exists(config_dir_path):
    print("Config directory does not exist:", config_dir_path)
else:
    print("Config directory exists.")

def load_api_models(filename="api_models_list.json"):
    full_path = get_config_path(os.path.join("../configs", filename))
    # Load the JSON data from supported api models json
    with open(full_path, 'r') as file:
        data = json.load(file)
    return data

def load_models_dict_json(filename="models_dict.json"):
    full_path = get_config_path(os.path.join("../configs", filename))
    # Load the JSON data from the file
    with open(full_path, 'r') as file:
        data = json.load(file)
    return data

def load_local_models_dict_json(filename="local_models_dict.json"):
    full_path = get_config_path(os.path.join("../configs", filename))
    # Load the JSON data from the file
    with open(full_path, 'r') as file:
        data = json.load(file)
    return data

def load_dataset_category_dict(dataset_file_path):
    if 'HEx-PHI' in dataset_file_path:
        full_path = get_config_path(os.path.join("../configs", "hex_phi_file_dict.json"))
        with open(full_path, 'r') as file:
            dict = json.load(file)
            filename = dataset_file_path.split('/')[-1]
            filename_without_extension = filename.replace('.csv', '')  # Remove the .csv extension
            return dict[filename_without_extension]
    else:
        return ""

def load_woke_template(file_path="woke_templates.json", name="woke-template-v1"):
    full_path = get_config_path(os.path.join("../configs", file_path))
    with open(full_path, "r") as file:
        templates = json.load(file)
    return templates[name]
