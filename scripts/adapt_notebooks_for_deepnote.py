#!/usr/bin/env python3
"""
Script para adaptar todos os notebooks para Deepnote.
Substitui paths do Google Drive por paths locais.
"""

import os
import re
from pathlib import Path

def adapt_notebook_content(content):
    """Adapta o conteúdo de um notebook para Deepnote"""
    
    # Substituir paths do Google Drive por paths locais
    replacements = [
        # Paths principais
        ("/content/drive/MyDrive/mhealth-data", "./data"),
        ("'/content/drive/MyDrive/mhealth-data'", "'./data'"),
        ('"/content/drive/MyDrive/mhealth-data"', '"./data"'),
        
        # Paths específicos
        ("/content/drive/MyDrive/mhealth-data/raw", "./data/raw"),
        ("/content/drive/MyDrive/mhealth-data/processed", "./data/processed"),
        ("/content/drive/MyDrive/mhealth-data/models", "./data/models"),
        ("/content/drive/MyDrive/mhealth-data/results", "./data/results"),
        
        # Referências ao setup
        ("00_colab_setup.ipynb", "00_deepnote_setup.ipynb"),
        ("run 00_colab_setup.ipynb first", "run 00_deepnote_setup.ipynb first"),
        
        # Textos descritivos
        ("Google Drive", "Deepnote storage"),
        ("salva dados processados no Google Drive", "salva dados processados no storage local do Deepnote"),
        ("Salva modelos e resultados", "Salva modelos e resultados no storage local"),
        
        # Instruções de upload
        ("download", "upload"),
        ("place it in the raw directory", "upload to the ./data/raw/ directory"),
        ("Use Deepnote's file upload feature", "Use Deepnote's file upload feature"),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    return content

def adapt_notebook_file(file_path):
    """Adapta um arquivo de notebook"""
    print(f"Adapting {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    adapted_content = adapt_notebook_content(content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(adapted_content)
    
    print(f"✅ Adapted {file_path}")

def main():
    """Adapta todos os notebooks"""
    notebooks_dir = Path("notebooks")
    
    if not notebooks_dir.exists():
        print("❌ Notebooks directory not found")
        return
    
    # Lista de notebooks para adaptar (exceto o setup que já foi adaptado)
    notebook_files = [
        "01_preprocess_sleep_edf.ipynb",
        "02_preprocess_wesad.ipynb", 
        "03_train_baseline.ipynb",
        "04_train_dp.ipynb",
        "05_train_fl.ipynb",
        "06_analysis.ipynb"
    ]
    
    print("="*70)
    print("ADAPTING NOTEBOOKS FOR DEEPNOTE")
    print("="*70)
    
    for notebook_file in notebook_files:
        file_path = notebooks_dir / notebook_file
        if file_path.exists():
            adapt_notebook_file(file_path)
        else:
            print(f"⚠️  {notebook_file} not found")
    
    print("\n" + "="*70)
    print("ADAPTATION COMPLETE!")
    print("="*70)
    print("All notebooks have been adapted for Deepnote.")
    print("Key changes:")
    print("- Google Drive paths → Local ./data/ paths")
    print("- Setup notebook reference updated")
    print("- Upload instructions updated for Deepnote")

if __name__ == "__main__":
    main()

