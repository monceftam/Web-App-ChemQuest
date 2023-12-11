#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, shapiro, kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.mixture import GaussianMixture
from chembl_webresource_client.new_client import new_client
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from chembl_webresource_client.new_client import new_client
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import mannwhitneyu


# In[2]:


def get_target_data(target_query):
    target = new_client.target
    return pd.DataFrame.from_dict(target.search(target_query))


# In["]:


def get_activity_data(chembl_id, activity_type="IC50"):
    activity = new_client.activity
    return pd.DataFrame.from_dict(activity.filter(target_chembl_id=chembl_id).filter(standard_type=activity_type))


# In[4]:


def preprocess_activity_data(df):
    df = df[df.standard_value.notna()]
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce').dropna()
    df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
    return df


# In[5]:


def remove_lower_outliers(df, column):
    lower_bound = df[column].quantile(0.25) - 1.5 * (df[column].quantile(0.75) - df[column].quantile(0.25))
    return df[df[column] >= lower_bound]


# In[6]:


def classify_compounds(df, column):
    gmm = GaussianMixture(n_components=2, random_state=0).fit(df[[column]])
    active_threshold = np.max(gmm.means_)
    df['bioactivity_class'] = df[column].apply(lambda x: 'active' if x >= active_threshold else 'inactive')
    return df, active_threshold


# In[7]:


def display_classification(df, pIC50_cutoff):
    sns.histplot(df['pIC50'], bins=30, kde=False)
    plt.axvline(x=pIC50_cutoff, color='green', linestyle='--', label='Active Cutoff')
    plt.xlabel('pIC50')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# In[8]:


def logistic_regression_analysis(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[['pIC50']], df['bioactivity_class'].map({'active': 1, 'inactive': 0}), test_size=0.2, random_state=42
    )
    model = LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC AUC score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    return model


# In[9]:


# Main execution flow
targets_df = get_target_data("INSR")
selected_target_id = targets_df.iloc[3]['target_chembl_id']
activities_df = get_activity_data(selected_target_id)

preprocessed_df = preprocess_activity_data(activities_df)
filtered_df = remove_lower_outliers(preprocessed_df, 'pIC50')
classified_df, active_cutoff = classify_compounds(filtered_df, 'pIC50')
display_classification(classified_df, active_cutoff)


# In[10]:


# Perform logistic regression analysis
lr_model = logistic_regression_analysis(classified_df)
# Keep only compounds classified as 'active'
active_compounds_df = active_compounds_df[active_compounds_df['bioactivity_class'] == 'active']

# Save the active compounds DataFrame to CSV if needed
active_compounds_df.to_csv('active_insr_compounds.csv', index=False)

print(active_compounds_df.head(10))


# In[11]:


def process_compounds(active_compounds_df):
    # Define Lipinski's rule
    def lipinski_rule(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            return (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
        else:
            return False

    # Apply Lipinski's rule and other properties
    active_compounds_df['Lipinski_Rule_Passed'] = active_compounds_df['canonical_smiles'].apply(lipinski_rule)

    # Filter compounds that pass Lipinski's rule
    filtered_df = active_compounds_df[active_compounds_df['Lipinski_Rule_Passed']]

    return filtered_df

def display_top_compounds_by_ligand_efficiency(filtered_df, n=20):
    # Ensure 'ligand_efficiency' is a numeric column
    filtered_df['ligand_efficiency'] = pd.to_numeric(filtered_df['ligand_efficiency'], errors='coerce')
    
    # Sort the DataFrame by 'ligand_efficiency' in descending order
    top_compounds = filtered_df.sort_values(by='ligand_efficiency', ascending=False).head(n)
    
    # Define columns to display
    columns_to_display = ['action_type', 'document_journal', 'molecule_chembl_id', 'activity_properties', 
                          'assay_chembl_id', 'assay_type', 'assay_variant_accession', 'assay_variant_mutation', 
                          'target_organism', 'target_pref_name', 'parent_molecule_chembl_id', 
                          'molecule_pref_name', 'ligand_efficiency', 'document_year']

    # Display the top compounds
    print(top_compounds[columns_to_display])


# In[12]:


# Example usage
active_compounds_df = pd.read_csv('active_insr_compounds.csv') # Replace with your file path
filtered_df = process_compounds(active_compounds_df)
display_top_compounds_by_ligand_efficiency(filtered_df)







