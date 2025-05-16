import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_uniprot_xml_biotite(up_results):
    """
    Extract protein annotations from biotite UniProt XML results.
    
    Args:
        up_results: Dictionary from biotite.database.uniprot.fetch with format="xml"
        
    Returns:
        DataFrame with annotations
    """
    results = []
    
    # Define UniProt XML namespace
    ns = {'up': 'http://uniprot.org/uniprot'}
    
    for uniprot_id, xml_text in tqdm(up_results.items(), desc="Parsing UniProt XML"):
        # Parse XML string
        try:
            root = ET.fromstring(xml_text)
            entry = root.find('.//up:entry', ns)
            
            if entry is None:
                print(f"Warning: No entry found for {uniprot_id}")
                continue
                
            protein_info = {
                "uniprot_id": uniprot_id,
                "protein_name": "",
                "gene_names": [],
                "protein_class": [],
                "subcellular_location": [],
                "domains": [],
                "regions": [],
                "functions": []
            }
            
            # Extract protein name
            protein_elem = entry.find('./up:protein', ns)
            if protein_elem is not None:
                rec_name = protein_elem.find('./up:recommendedName', ns)
                if rec_name is not None:
                    full_name = rec_name.find('./up:fullName', ns)
                    if full_name is not None and full_name.text:
                        protein_info["protein_name"] = full_name.text
            
            # Extract gene names
            gene_elems = entry.findall('./up:gene/up:name', ns)
            for gene_elem in gene_elems:
                if gene_elem.text:
                    protein_info["gene_names"].append(gene_elem.text)
            
            # Extract keywords (protein class)
            keyword_elems = entry.findall('./up:keyword', ns)
            for kw_elem in keyword_elems:
                if kw_elem.text:
                    protein_info["protein_class"].append(kw_elem.text)
            
            # Extract subcellular location
            comment_elems = entry.findall('./up:comment', ns)
            for comment in comment_elems:
                comment_type = comment.get('type')
                
                if comment_type == 'subcellular location':
                    loc_elems = comment.findall('.//up:location', ns)
                    for loc in loc_elems:
                        if loc.text and loc.text not in protein_info["subcellular_location"]:
                            protein_info["subcellular_location"].append(loc.text)
                
                elif comment_type == 'function':
                    text_elems = comment.findall('./up:text', ns)
                    for text in text_elems:
                        if text.text:
                            protein_info["functions"].append(text.text)
            
            # Extract regions of interest
            feature_elems = entry.findall('./up:feature', ns)
            for feature in feature_elems:
                feature_type = feature.get('type')
                
                if feature_type == 'region of interest':
                    location = feature.find('./up:location', ns)
                    if location is not None:
                        begin = location.find('./up:begin', ns)
                        end = location.find('./up:end', ns)
                        
                        if begin is not None and end is not None:
                            start_pos = int(begin.get('position'))
                            end_pos = int(end.get('position'))
                            description = feature.get('description', '')
                            
                            feature_info = {
                                "name": description,
                                "start": start_pos,
                                "end": end_pos
                            }

                            protein_info["regions"].append(feature_info)

            # Extract pfam domains
            pfam_elems = entry.findall('./up:dbReference[@type="Pfam"]', ns)
            for pfam in pfam_elems:
                pfam_id = pfam.get('id')
                if pfam_id:
                    protein_info["domains"].append(pfam_id)
                    protein_info["domains"] = list(set(protein_info["domains"]))  # Remove duplicates
            
            results.append(protein_info)
            
        except ET.ParseError as e:
            print(f"Error parsing XML for {uniprot_id}: {e}")
    
    # Convert lists to comma-separated strings for better DataFrame handling
    result_df = pd.DataFrame(results)
    
    # Convert gene names list to string
    result_df['gene_names'] = result_df['gene_names'].apply(lambda x: ', '.join(x) if x else '')
    
    return result_df

# Example usage:
from biotite.database import uniprot
protein_ids = ["P01308", "P04637", "P42345"]
up_results = uniprot.fetch(protein_ids, format="xml")
up_results = {pid: upr.read() for pid, upr in zip(protein_ids, up_results)}

for pid, upr in up_results.items():
    with open(f"{pid}.xml", "w") as f:
        f.write(upr)

annotations_df = parse_uniprot_xml_biotite(up_results)
annotations_df.to_csv("uniprot_annotations.csv", index=False)