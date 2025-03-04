import xml.etree.ElementTree as ET
import os


def extract_rpc_data(xml_file, output_file):
    """
    Extracts RPC data from the given XML file and writes it to the output file.

    Args:
        xml_file (str): Path to the input XML file.
        output_file (str): Path to the output text file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the RPC data in the XML file
    rpc_data = root.find("RPC")
    if rpc_data is None:
        print(f"No RPC data found in the XML file: {xml_file}")
        return

    # Write the RPC data to the output file
    with open(output_file, "w") as f:
        for child in rpc_data:
            f.write(f"{child.tag}: {child.text.strip()}\n")


if __name__ == "__main__":
    # Define the input and output folders
    input_folder = "input_data"
    output_folder = "rpc_extracted_data"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all XML files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".xml"):
            xml_file = os.path.join(input_folder, filename)
            output_file = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}_rpc_data.txt"
            )
            # Extract RPC data from the XML file and write it to the output file
            extract_rpc_data(xml_file, output_file)
            print(f"RPC data has been extracted to {output_file}")
