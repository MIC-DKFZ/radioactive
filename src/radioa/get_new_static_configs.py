import sys
from pathlib import Path


def modify_config(template_path, models):
    for model in models:
        # Define the special cases for model names
        special_titles = {"sammed3d_turbo": "SAMMED3DTurbo", "sammed3d_turbonorm": "SAMMED3DTurboNORM"}

        # Check for special title cases
        if model in special_titles:
            title_model = special_titles[model]
        else:
            title_model = model.upper()

        # Read the template file
        with open(template_path, "r") as file:
            template_lines = file.readlines()

        for dataset_number in range(1, 12):  # From D1 to D11
            lines = template_lines[:]  # Copy template lines for each file modification

            # Process to uncomment the specified model
            model_found = False
            for i in range(len(lines)):
                if lines[i].strip() == "models:":
                    model_found = True
                elif model_found and lines[i].strip().startswith('# - "' + model + '"'):
                    lines[i] = lines[i].replace("# -", "-")  # Uncomment the line
                    break

            # Process to uncomment the specified dataset
            dataset_count = 0
            dataset_found = False
            for i in range(len(lines)):
                if lines[i].strip() == "datasets:":
                    dataset_found = True
                elif dataset_found and lines[i].strip().startswith("# - identifier:"):
                    dataset_count += 1
                    if dataset_count == dataset_number:
                        # Uncomment the next three lines related to the dataset block
                        for j in range(3):  # Uncomment exactly three lines
                            if i < len(lines) and "#" in lines[i]:
                                lines[i] = lines[i].replace("# ", "", 1)
                            i += 1
                        break

            # Determine the new file name
            new_file_path = Path(template_path).parent / f"static_prompt_{title_model}_D{dataset_number}.yaml"

            # Write the modified config to a new file
            with open(new_file_path, "w") as file:
                file.writelines(lines)


if __name__ == "__main__":
    template_path = "configs/template_static.yaml"
    models = ["sam"]  # Update as needed
    modify_config(template_path, models)
