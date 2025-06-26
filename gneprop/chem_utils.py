from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm


def compute_filters(mol):
    results = dict()

    molecular_weight = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_bond_donor = Descriptors.NumHDonors(mol)
    h_bond_acceptors = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(mol)
    molar_refractivity = Chem.Crippen.MolMR(mol)
    topological_surface_area_mapping = Chem.QED.properties(mol).PSA
    formal_charge = Chem.rdmolops.GetFormalCharge(mol)
    heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)
    num_of_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
    tpsa = Descriptors.TPSA(mol)

    results["lipinski"] = int(molecular_weight <= 500) + int(logp <= 5) + int(h_bond_donor <= 5) + int(h_bond_acceptors <= 10) >=4 # >= 3
    results["lipinski_1"] = int(molecular_weight <= 500) + int(logp <= 5) + int(h_bond_donor <= 5) + int(h_bond_acceptors <= 10) >=3 # >= 3

    results['ghose'] = 160 <= molecular_weight <= 480 and 0.4 <= logp <= 5.6 and 20 <= number_of_atoms <= 70 and 40 <= molar_refractivity <= 130
    results['veber'] = rotatable_bonds <= 10 and topological_surface_area_mapping <= 140
    results['rule3'] = molecular_weight <= 300 and logp <= 3 and h_bond_donor <= 3 and h_bond_acceptors <= 3 and rotatable_bonds <= 3
    results['reos'] = 200 <= molecular_weight <= 500 and -5 <= logp <= 5 and 0 <= h_bond_donor <= 5 and 0 <= h_bond_acceptors <= 10 and -2 <= formal_charge <= 2 and 0 <= rotatable_bonds <= 8 and 15 <= heavy_atoms <= 50
    results['drug-like'] = molecular_weight < 400 and num_of_rings > 0 and rotatable_bonds < 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and logp < 5
    results['antibiotic-like'] = 250 <= molecular_weight <= 550 and tpsa >= 80 and logp <= 2.0 and h_bond_donor >= 2 and h_bond_acceptors >= 4 and rotatable_bonds >= 2

    results['all'] = all(results.values())

    return results


def compute_filters_smiles(smiles):
    return [compute_filters(Chem.MolFromSmiles(s)) for s in tqdm(smiles, position=0, leave=True)]


def filters_statistics(list_filters):
    filters_key = list_filters[0].keys()
    for fk in filters_key:
        num_trues = sum([int(i[fk]) for i in list_filters])
        print(f'{fk}: {num_trues} positives ({(num_trues/len(list_filters))*100:.2f}%)')


