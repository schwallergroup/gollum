from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, rdMolDescriptors
import numpy as np
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import pandas as pd
import os
from pathlib import Path

def fingerprints(smiles, bond_radius=3, nBits=2048):
    """
    Get Morgan fingerprints for a list of SMILES strings.
    """
    rdkit_mols = [MolFromSmiles(smile) for smile in smiles]
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
        for mol in rdkit_mols
    ]
    return np.asarray(fingerprints)


# auxiliary function to calculate the fragment representation of a molecule
def fragments(smiles):
    # descList[115:] contains fragment-based features only
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
    # Update: in the new RDKit version the indices are [124:]
    fragments = {d[0]: d[1] for d in Descriptors.descList[124:]}
    frags = np.zeros((len(smiles), len(fragments)))
    for i in range(len(smiles)):
        mol = MolFromSmiles(smiles[i])
        try:
            features = [fragments[d](mol) for d in fragments]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        frags[i, :] = features

    return frags



def mqn_features(smiles):
    """
    Builds molecular representation as a vector of Molecular Quantum Numbers.
    :param reaction_smiles: list of molecular smiles
    :type reaction_smiles: list
    :return: array of mqn featurised molecules
    """
    molecules = [MolFromSmiles(smile) for smile in smiles]
    mqn_descriptors = [
        rdMolDescriptors.MQNs_(molecule) for molecule in molecules
    ]
    return np.asarray(mqn_descriptors)


def chemberta_features(smiles):
    # any model weights from the link above will work here
    model = AutoModelWithLMHead.from_pretrained(
        "seyonec/ChemBERTa-zinc-base-v1"
    )
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenized_smiles = [
        tokenizer(smile, return_tensors="pt") for smile in smiles
    ]
    outputs = [
        model(
            input_ids=tokenized_smile["input_ids"],
            attention_mask=tokenized_smile["attention_mask"],
            output_hidden_states=True,
        )
        for tokenized_smile in tokenized_smiles
    ]
    embeddings = torch.cat(
        [output["hidden_states"][0].sum(axis=1) for output in outputs], axis=0
    )
    return embeddings.detach().numpy()

def cddd(smiles):
    current_path = os.getcwd()
    os.chdir(Path(os.path.abspath(__file__)).parent)
    cddd = pd.read_csv(
        "precalculated_featurisation/cddd_additives_descriptors.csv"
    )
    cddd_array = np.zeros((cddd.shape[0], 512))
    for i, smile in enumerate(smiles):
        row = cddd[cddd["smiles"] == smile][cddd.columns[3:]].values
        cddd_array[i] = row
    os.chdir(current_path)
    return cddd_array


def xtb(smiles):
    current_path = os.getcwd()
    os.chdir(Path(os.path.abspath(__file__)).parent)
    xtb = pd.read_csv("precalculated_featurisation/xtb_qm_descriptors_2.csv")
    xtb_array = np.zeros((xtb.shape[0], len(xtb.columns[:-2])))
    for i, smile in enumerate(smiles):
        row = xtb[xtb["Additive_Smiles"] == smile][xtb.columns[:-2]].values
        xtb_array[i] = row
    os.chdir(current_path)
    return xtb_array