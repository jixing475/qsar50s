#!/Users/zero/anaconda3/envs/reticulate/bin/python

# use
# input: df with a SMILES column 
# output: df with description


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
import pandas as pd
import numpy as np
from useful_rdkit_utils import mol2numpy_fp
from padelpy import from_smiles



#------------------------------------------------
# function
#------------------------------------------------

# MWT 
# LGP
# HBD
# HBA
# RTB
# TPSA



def mol_to_TPSA(mol):
    try:
        TPSA=Chem.Descriptors.TPSA(mol)
        return TPSA
    except:
        return None

def mol_to_MWT(mol):
    try:
        MWT=Chem.Descriptors.ExactMolWt(mol)
        return MWT
    except:
        return None
  
def mol_to_LGP(mol):
    try:
        LGP=Chem.Descriptors.MolLogP(mol)
        return LGP
    except:
        return None
 

def mol_to_HBD(mol):
    try:
        HBD=Chem.Descriptors.NumHDonors(mol)
        return HBD
    except:
        return None


def mol_to_HBA(mol):
    try:
        HBA=Chem.Descriptors.NumHAcceptors(mol)
        return HBA
    except:
        return None

def mol_to_RTB(mol):
    try:
        RTB=Chem.Descriptors.NumRotatableBonds(mol)
        return RTB
    except:
        return None
        
def canonic_smiles(mol):
    try:
      canonic_smiles = Chem.MolToSmiles(mol)
      return canonic_smiles
    except:
      return None

def dwa_smiles(mol):
    try:
        # 移除分子中的氢原子
        mol = Chem.RemoveHs(mol)
        dwa_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return dwa_smiles
    except:
        return None

def df_add_dwa_SMILES(df):
    df['dwa_SMILES'] = df.ROMol.map(dwa_smiles)  # canonic_smiles
    return df


def mol_to_QED(mol):
    """
    Computes RDKit's QED score
    """
    try:
      QED = qed(mol)
      return QED
    except:
      return None

def mol_to_SA(mol):
    """
    Computes RDKit's SA score
    """
    try:
      SA = sascorer.calculateScore(mol)
      return SA
    except:
      return None


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    try:
      n_rings = mol.GetRingInfo().NumRings()
      return n_rings
    except:
      return None

# pass filter ==============================


def rule_of_five(m):
  if m is None:
    return None
  # Calculate rule of five chemical properties
  MW = Descriptors.ExactMolWt(m)
  HBA = Descriptors.NumHAcceptors(m)
  HBD = Descriptors.NumHDonors(m)
  LogP = Descriptors.MolLogP(m)
  # Rule of five conditions
  conditions = [MW <= 500, HBA <= 10, HBD <= 5, LogP <= 5]
  # Create pandas row for conditions results with values and information whether rule of five is violated
  # return 'pass' if conditions.count(True) >= 3 else 'no'
  return True if conditions.count(True) >= 3 else False


def rule_of_3(m):
  """
  MW<=300
  LogP > -3 and LogP < 3
  HBA <= 3
  HBD <= 3
  tPSA <= 60,
  Rotatable bonds <=3
  """
  MW = Descriptors.ExactMolWt(m)
  HBA = Descriptors.NumHAcceptors(m)
  HBD = Descriptors.NumHDonors(m)
  LogP = Descriptors.MolLogP(m)
  RTB= Descriptors.NumRotatableBonds(m)
  TPSA= Descriptors.TPSA(m)
  # Rule of five conditions
  conditions = [MW <= 300, HBA <= 3, HBD <= 3, LogP <= 3, LogP >= -3, TPSA <= 60, RTB <= 3]
  # Create pandas row for conditions results with values and information whether rule of five is violated
  # return 'pass' if conditions.count(True) >= 3 else 'no'
  return True if conditions.count(True) >= 7 else False


def rule_of_2(m):
  MW = Descriptors.ExactMolWt(m)
  HBA = Descriptors.NumHAcceptors(m)
  HBD = Descriptors.NumHDonors(m)
  LogP = Descriptors.MolLogP(m)
  RTB= Descriptors.NumRotatableBonds(m)
  TPSA= Descriptors.TPSA(m)
  # Rule of five conditions
  conditions = [MW < 300, HBA == 4, HBD == 2, LogP < 2]
  return True if conditions.count(True) == 4 else False



def df_add_rule_of_2_3_5(df):
  df['pass_rule_of_2'] = df.ROMol.map(rule_of_2)
  df['pass_rule_of_3'] = df.ROMol.map(rule_of_3)
  df['pass_rule_of_5'] = df.ROMol.map(rule_of_five)
  return df




def df_to_descriptors(df):
  # smiles to mol object ==============================
  PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='SMILES')
 # mole to descriptors ==============================
  df['CANONIC_SMILES'] = df.ROMol.map(canonic_smiles)# canonic_smiles
  df['MWT'] = df.ROMol.map(mol_to_MWT)# MWT 
  df['LGP'] = df.ROMol.map(mol_to_LGP)# LGP
  df['HBD'] = df.ROMol.map(mol_to_HBD)# HBD
  df['HBA'] = df.ROMol.map(mol_to_HBA)# HBA
  df['RTB'] = df.ROMol.map(mol_to_RTB)# RTB
  df['TPSA'] = df.ROMol.map(mol_to_TPSA)# TPSA
  df['QED'] = df.ROMol.map(mol_to_QED)# QED
  df['SA'] = df.ROMol.map(mol_to_SA)# SA
  df['n_rings'] = df.ROMol.map(get_n_rings)# n_rings
  df['pass_mcf_pains'] = df.ROMol.map(pass_mcf_pains)# 是否通过 MCF
  df['pass_rule_of_5'] = df.ROMol.map(rule_of_five)
  
  # remove mol object columns ==============================
  # df = df.drop(columns=["ROMol"])
  return df


def df_ROMol_to_descriptors(df):
 # mole to descriptors ==============================
  df['CANONIC_SMILES'] = df.ROMol.map(canonic_smiles)# canonic_smiles
  df['MWT'] = df.ROMol.map(mol_to_MWT)# MWT 
  df['LGP'] = df.ROMol.map(mol_to_LGP)# LGP
  df['HBD'] = df.ROMol.map(mol_to_HBD)# HBD
  df['HBA'] = df.ROMol.map(mol_to_HBA)# HBA
  df['RTB'] = df.ROMol.map(mol_to_RTB)# RTB
  df['TPSA'] = df.ROMol.map(mol_to_TPSA)# TPSA
  df['QED'] = df.ROMol.map(mol_to_QED)# QED
  df['SA'] = df.ROMol.map(mol_to_SA)# SA
  df['n_rings'] = df.ROMol.map(get_n_rings)# n_rings
  df['pass_mcf_pains'] = df.ROMol.map(pass_mcf_pains)# 是否通过 MCF
  df['pass_rule_of_5'] = df.ROMol.map(rule_of_five)
  
  # remove mol object columns ==============================
  # df = df.drop(columns=["ROMol"])
  return df

def df_add_CANONIC_SMILES(df):
  df['CANONIC_SMILES'] = df.ROMol.map(canonic_smiles)# canonic_smiles
  return df



def df_add_ROMol(df):
  # smiles to mol object ==============================
  PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='SMILES')
 # mole to descriptors ==============================
  df['CANONIC_SMILES'] = df.ROMol.map(canonic_smiles)# canonic_smiles
  return df

# 加的 H 没有坐标信息
def add_H_coords(mol):
    """
    add H with coords
    """
    try:
      mol = Chem.AddHs(mol, addCoords=True)
      return mol
    except:
      return None


def add_H_ROMol(df):
  df["ROMol"] = df["ROMol"].map(add_H_coords)
  return df

def check_mol_has_substructure(mol, substructure_mol):
  """Checks if rdkit.Mol has substructure.
  Args:
    mol : rdkit.Mol, representing query
    substructure_mol: rdkit.Mol, representing substructure family
  Returns:
    Boolean, True if substructure found in molecule.
  """
  return mol.HasSubstructMatch(substructure_mol)

def df_check_mol_has_substructure(df, SMARTS):
  substructure_mol = Chem.MolFromSmarts(SMARTS)
  df["has_substructure"] = df.apply(lambda x: check_mol_has_substructure(x['ROMol'], substructure_mol), axis=1)
  return df



# 检查是否至少含有多个子结构中的一个
def check_mol_has_substructures(mol, substructure_mols):
  if mol is None:
    return None
  if any(mol.HasSubstructMatch(smarts) for smarts in substructure_mols):
    return True
  return False 

def df_check_mol_has_substructures(df, SMARTS_list):
  substructure_mols = [Chem.MolFromSmarts(x) for x in SMARTS_list]
  df["has_substructure"] = df.apply(lambda x: check_mol_has_substructures(x['ROMol'], substructure_mols), axis=1)
  return df



# 在 df 中计算 RMSD

def two_mol_inplace_rmsd(ref, target):
    # ---- Read reference and calculate RMSD ---- #
    """
    Parameters
    ----------
    ref: mol obj to the reference structure 
    target: mol to target poses 
    """
    # ref = Chem.MolFromMol2File(ref)
    # ref.SetProp('_Name', 'Ref')
    #
    r = rdFMCS.FindMCS([ref, target])
    #
    a = ref.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    b = target.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    amap = list(zip(a, b))
    #
    distances = []
    for atomA, atomB in amap:
        pos_A = ref.GetConformer().GetAtomPosition(atomA)
        pos_B = target.GetConformer().GetAtomPosition(atomB)
        coord_A = np.array((pos_A.x, pos_A.y, pos_A.z))
        coord_B = np.array((pos_B.x, pos_B.y, pos_B.z))
        dist_numpy = np.linalg.norm(coord_A - coord_B)
        distances.append(dist_numpy)
    #
    rmsd = math.sqrt(1 / len(distances) * sum([i * i for i in distances]))
    #
    return rmsd
  
def df_add_inplace_rmsd(df, ref_mol_path):
    ref = Chem.MolFromMol2File(ref_mol_path, sanitize=True)
    ref.SetProp('_Name', 'Ref')
    df["rmsd_inplace"] = df.apply(lambda x: two_mol_inplace_rmsd(x['ROMol'], ref), axis=1)
    return df


def mol_to_maccs_fp(mol):
    """
    Computes MACCS keys for a molecule.
    """
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    except:
        return None


def mol_to_morgan_fp(mol, radius=2, nBits=1024):
    """
    Computes Morgan fingerprint for a molecule.
    """
    try:
        return mol2numpy_fp(mol, radius=radius, nBits=nBits)
    except:
        return None

def df_add_fp(df, fp_type='morgan', **kwargs):
    """
    Adds molecular fingerprint columns to DataFrame.
    
    Parameters
    ----------
    df : pandas DataFrame
        Input DataFrame containing RDKit mol objects in 'ROMol' column
    fp_type : str
        Type of fingerprint to generate. Options: 'morgan', 'maccs'
    **kwargs:
        Additional keyword arguments for specific fingerprint types.
        For 'morgan': radius (default 2), nBits (default 1024)
        
    Returns
    -------
    df : pandas DataFrame
        DataFrame with added fingerprint columns
    """
    if fp_type == 'morgan':
        radius = kwargs.get('radius', 2)
        nBits = kwargs.get('nBits', 1024)
        fp_func = lambda x: mol_to_morgan_fp(x, radius=radius, nBits=nBits)
        fp_cols = [f'morgan_{i}' for i in range(nBits)]
        fp_data = df['ROMol'].apply(fp_func)
        fp_df = pd.DataFrame(fp_data.tolist(), columns=fp_cols, index=df.index)
    elif fp_type == 'maccs':
        fp_func = mol_to_maccs_fp
        fp_cols = [f'maccs_{i}' for i in range(167)] # MACCS keys are 167 bits
        fp_data = df['ROMol'].apply(fp_func)
        fp_df = pd.DataFrame(fp_data.tolist(), columns=fp_cols, index=df.index)
    elif fp_type == 'pubchem':
        smiles_list = df['ROMol'].apply(canonic_smiles).tolist()
        fingerprints = from_smiles(smiles_list, fingerprints=True, descriptors=False, threads=1, timeout=600)
        fp_df = pd.DataFrame(fingerprints)
        fp_df = fp_df.reindex(df.index)
    elif fp_type == 'padel_desc':
        smiles_list = df['ROMol'].apply(canonic_smiles).tolist()
        descriptors = from_smiles(smiles_list, fingerprints=False, descriptors=True, threads=1, timeout=600)
        fp_df = pd.DataFrame(descriptors)
        fp_df = fp_df.reindex(df.index)
    else:
        raise ValueError("fp_type must be 'morgan', 'maccs', 'pubchem', or 'padel_desc'")

    df = pd.concat([df, fp_df], axis=1)
    return df

