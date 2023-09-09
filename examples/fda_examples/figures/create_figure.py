import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Get the absolute path of the currently executing script
grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(grandparent_dir)
from pickle_data.pickle_functions import load_pickle
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from io import BytesIO
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Create figures using pickle files')
parser.add_argument('-f', type=str, default='', nargs='+', help='list of pickle files')
parser.add_argument("--syncs", action="store_true")
parser.add_argument("--basis", type=str, default="exper_type")
args = parser.parse_args()

def save_svg_with_description(figure, filename, description):
    """
    Save a matplotlib figure as an SVG with an embedded description.
    
    Parameters:
    - figure: Matplotlib figure to be saved.
    - filename: Desired filename for the SVG.
    - description: Text description to embed in the SVG.
    """
    # Check if the file exists and generate a new filename if necessary
    base_name, ext = os.path.splitext(filename)
    count = 0
    new_file = filename

    while os.path.exists(new_file):
        count += 1
        new_file = f"{base_name} ({count}){ext}"

    # Save the figure to a BytesIO object first
    f = BytesIO()
    figure.savefig(f, format='svg')
    plt.close(figure)

    # Parse the SVG data
    f.seek(0)
    tree = ET.parse(f)
    root = tree.getroot()

    # Convert description to a single string if it's a list
    if isinstance(description, list):
        description = ", ".join(description)

    # Add the description to the SVG
    desc = ET.SubElement(root, 'desc')
    desc.text = description

    # Save the modified SVG to a file
    with open(new_file, 'wb') as svg_out:
        svg_out.write(ET.tostring(root))
    
    print("Syncs figure saved in: " + new_file)


def syncs_figure(filenames, basis):

    file_data = []

    # Load the stored data
    for i in range(len(filenames)):
        file_data.append(load_pickle(filenames[i]))

    filename = "Syncs." + "on_" + basis + ".x" + str(len(filenames)) + ".svg"
    figure_file = os.path.dirname(__file__) + "/" + filename
    
    plt.figure()

    for i in range(len(file_data)):
        lib = file_data[i]
        plt.plot(lib["step"], lib["syncs"], label=str(lib[basis]))

    plt.xlabel('Step / Batch Count')
    plt.ylabel('Synchronizations')
    plt.title('Synchronizations per Batch Count')
    plt.axis('scaled')
    plt.legend()

    figure_file = save_svg_with_description(plt.gcf(), figure_file, filenames)

if args.syncs:
    syncs_figure(args.f, args.basis)