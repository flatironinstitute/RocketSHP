import re
import numpy as np
from biotite.database import uniprot

def extract_uniprot_info(uniprot_text):
    # Extract subcellular location
    subcellular_pattern = r"CC   -!- SUBCELLULAR LOCATION:(.*?)(?=CC   -!-|//)"
    subcellular_match = re.search(subcellular_pattern, uniprot_text, re.DOTALL)
    subcellular_location = subcellular_match.group(1).strip() if subcellular_match else "Not found"
    subcellular_location = re.sub(r"CC\s+", "", subcellular_location)
    
    # Extract function
    function_pattern = r"CC   -!- FUNCTION:(.*?)(?=CC   -!-|//)"
    function_match = re.search(function_pattern, uniprot_text, re.DOTALL)
    function = function_match.group(1).strip() if function_match else "Not found"
    function = re.sub(r"CC\s+", "", function)
    
    # Extract domains, regions and structural features
    features = []
    lines = uniprot_text.splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a feature line
        if line.startswith("FT   "):
            # Extract feature type and position
            parts = line[5:].strip().split(maxsplit=2)
            
            if len(parts) >= 2:
                feature_type = parts[0]
                position = parts[1]
                description = parts[2] if len(parts) > 2 else ""
                
                # Look for continuation lines
                i += 1
                while i < len(lines) and lines[i].startswith("FT                   "):
                    description += " " + lines[i][21:].strip()
                    i += 1
                i -= 1  # Adjust because we'll increment i at the end of the loop
                
                # Add the feature
                features.append({
                    "type": feature_type,
                    "position": np.arange(int(position.split("..")[0]), int(position.split("..")[-1]) + 1, dtype=int),
                    "description": description.strip()
                })
        
        i += 1
    
    # Categorize features
    domains_regions = [f for f in features if f["type"] in 
                      ["DOMAIN", "REGION", "MOTIF", "REPEAT", "SIGNAL", "PEPTIDE", "PROPEP"]]
    secondary_structure = [f for f in features if f["type"] in 
                          ["HELIX", "STRAND", "TURN"]]
    
    return {
        "subcellular_location": subcellular_location,
        "function": function,
        "domains_regions": domains_regions,
        "secondary_structure": secondary_structure
    }

FMT = "txt"
protein_ids = ["P01308", "P04637", "P42345"]  # Insulin, p53, mTOR
up_results = uniprot.fetch(protein_ids, format=FMT)
up_parsed = {}

for pid, upr in zip(protein_ids, up_results):
    with open(f"{pid}.{FMT}", "w") as f:
        info = extract_uniprot_info(upr.read())
    up_parsed[pid] = info

# print("SUBCELLULAR LOCATION:")
# print(info["subcellular_location"])

# print("\nFUNCTION:")
# print(info["function"])

# print("\nDOMAINS AND REGIONS:")
# for feature in info["domains_regions"]:
#     print(f"{feature['type']} ({feature['position']}): {feature['description']}")

# print("\nSECONDARY STRUCTURE:")
# for feature in info["secondary_structure"]:
#     print(f"{feature['type']} ({feature['position']}): {feature['description']}")