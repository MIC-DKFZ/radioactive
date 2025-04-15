import os

def generate_submit_script(input_folder, output_script_path):
    # Paths to be used in the submission script
    logs_path = "/dkfz/cluster/gpu/data/OE0441/c306h/logs"
    script_to_source = "~/radioactive.sh"
    experiments_runner = "/home/c306h/radioactive/universal-models/src/radioa/experiments_runner.py"

    # Start the script content
    script_content = "#!/bin/bash\n"

    # Iterate over all files in the input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".yaml") and not (file.startswith("test") or file.startswith("template")):
                # Full path to the configuration file
                config_path = os.path.join("/home/c306h/radioactive/universal-models", root, file)
                config_path = config_path.replace('/.', '')

                # Create the submission command
                submission_line = (
                    f"bsub -gpu num=1:j_exclusive=yes:gmem=10G -q gpu-lowprio -R tensorcore "
                    f"-o {logs_path}/%J.log -e {logs_path}/%J.err "
                    f"\". {script_to_source}  && python {experiments_runner} --config {config_path}\";"
                )

                # Append the line to the script content
                script_content += submission_line + "\n"

    # Write the script content to the output file
    with open(output_script_path, "w") as script_file:
        script_file.write(script_content)

    print(f"Submission script saved to {output_script_path}")

# Input folder containing the YAML configuration files
input_folder = "./configs"

# Output script file
output_script_path = "/home/c306h/cluster-data/submit_radioactive.sh"

# Generate the submit script
generate_submit_script(input_folder, output_script_path)
