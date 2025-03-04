import xml.etree.ElementTree as ET
import os


def extract_rpc_data(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    rpc_data = root.find("RPC")
    if rpc_data is None:
        print(f"No RPC data found in the XML file: {xml_file}")
        return

    with open(output_file, "w") as f:
        for child in rpc_data:
            f.write(f"{child.tag}: {child.text.strip()}\n")


if __name__ == "__main__":
    input_folder = "/home/ubuntu/code/ICEYE_baseline_Extractor/input_data"
    output_folder = "/home/ubuntu/code/ICEYE_baseline_Extractor/rpc_extracted_data"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".xml"):
            xml_file = os.path.join(input_folder, filename)
            output_file = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}_rpc_data.txt"
            )
            extract_rpc_data(xml_file, output_file)
            print(f"RPC data has been extracted to {output_file}")
