import csv
from Simulation_Gaussian_Broadening import diffraction_pattern

def run_diffraction_patterns(data_source, from_csv=False):
    num_grains_list = [10, 50, 100, 500, 1000, 5000]

    # Load data from CSV or use provided list
    if from_csv:
        material_data = []
        with open(data_source, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                material_type = row['material_type']
                crystal_structure = row['CrystalStructure']
                lattice_param = float(row['lattice_parameter'])
                material_data.append((material_type, crystal_structure, lattice_param))
    else:
        material_data = data_source

    # Loop through materials and grain numbers
    for material_type, crystal_structure, lattice_param in material_data:
        for num_grains in num_grains_list:
            diffraction_pattern(material_type, crystal_structure, num_grains, lattice_param)

# Example data if not using CSV
material_list = [
    ("Fe", "BCC", 0.2866),
    ("Cr", "BCC", 0.2885),
    ("W",  "BCC", 0.3165),
    ("Al", "FCC", 0.4049),
    ("Cu", "FCC", 0.3613),
    ("Au", "FCC", 0.4078),
    ("Ag", "FCC", 0.4086),
    ("Pt", "FCC", 0.3923),
]

# Example usage:
# run_diffraction_patterns("materials.csv", from_csv=True)
# or
run_diffraction_patterns(material_list, from_csv=False)

