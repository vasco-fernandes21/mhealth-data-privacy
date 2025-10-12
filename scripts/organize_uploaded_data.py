#!/usr/bin/env python3
"""
Script para reorganizar dados carregados no Deepnote.
Corrige a estrutura quando os arquivos são "achatados" durante o upload.
"""

import os
import shutil
from pathlib import Path

def organize_sleep_edf_files():
    """Organiza arquivos do Sleep-EDF"""
    print("🛌 Organizing Sleep-EDF files...")
    
    # Criar diretório se não existir
    sleep_dir = Path('./data/raw/sleep-edf')
    sleep_dir.mkdir(parents=True, exist_ok=True)
    
    # Lista de arquivos esperados do Sleep-EDF
    sleep_files = []
    for file in os.listdir('.'):
        if file.endswith('.rec') or file.endswith('.hyp') or file == 'RECORDS':
            sleep_files.append(file)
    
    if sleep_files:
        print(f"Found {len(sleep_files)} Sleep-EDF files:")
        for file in sleep_files:
            print(f"  📄 {file}")
            # Mover para a pasta correta
            shutil.move(file, sleep_dir / file)
        print(f"✅ Moved {len(sleep_files)} files to {sleep_dir}")
    else:
        print("⚠️  No Sleep-EDF files found")

def organize_wesad_files():
    """Organiza arquivos do WESAD"""
    print("😰 Organizing WESAD files...")
    
    # Criar diretório se não existir
    wesad_dir = Path('./data/raw/wesad')
    wesad_dir.mkdir(parents=True, exist_ok=True)
    
    # Lista de arquivos esperados do WESAD
    wesad_files = []
    for file in os.listdir('.'):
        if file.endswith('.pkl') and file.startswith('S'):
            wesad_files.append(file)
    
    if wesad_files:
        print(f"Found {len(wesad_files)} WESAD files:")
        for file in wesad_files:
            print(f"  📄 {file}")
            # Mover para a pasta correta
            shutil.move(file, wesad_dir / file)
        print(f"✅ Moved {len(wesad_files)} files to {wesad_dir}")
    else:
        print("⚠️  No WESAD files found")

def check_current_structure():
    """Verifica a estrutura atual"""
    print("🔍 Checking current directory structure...")
    
    current_files = os.listdir('.')
    print(f"Files in current directory: {len(current_files)}")
    
    # Categorizar arquivos
    sleep_files = [f for f in current_files if f.endswith('.rec') or f.endswith('.hyp') or f == 'RECORDS']
    wesad_files = [f for f in current_files if f.endswith('.pkl') and f.startswith('S')]
    other_files = [f for f in current_files if f not in sleep_files and f not in wesad_files]
    
    print(f"\n📊 File categorization:")
    print(f"  Sleep-EDF files: {len(sleep_files)}")
    print(f"  WESAD files: {len(wesad_files)}")
    print(f"  Other files: {len(other_files)}")
    
    if sleep_files:
        print(f"\n🛌 Sleep-EDF files found:")
        for file in sleep_files[:5]:  # Mostrar apenas os primeiros 5
            print(f"    {file}")
        if len(sleep_files) > 5:
            print(f"    ... and {len(sleep_files) - 5} more")
    
    if wesad_files:
        print(f"\n😰 WESAD files found:")
        for file in wesad_files[:5]:  # Mostrar apenas os primeiros 5
            print(f"    {file}")
        if len(wesad_files) > 5:
            print(f"    ... and {len(wesad_files) - 5} more")
    
    return sleep_files, wesad_files

def main():
    """Função principal"""
    print("="*70)
    print("ORGANIZING UPLOADED DATA")
    print("="*70)
    
    # Verificar estrutura atual
    sleep_files, wesad_files = check_current_structure()
    
    if not sleep_files and not wesad_files:
        print("\n❌ No dataset files found in current directory")
        print("Make sure you're in the right directory and files are uploaded")
        return
    
    # Organizar arquivos
    if sleep_files:
        organize_sleep_edf_files()
    
    if wesad_files:
        organize_wesad_files()
    
    # Verificar estrutura final
    print("\n" + "="*70)
    print("FINAL STRUCTURE")
    print("="*70)
    
    if os.path.exists('./data/raw/sleep-edf'):
        sleep_count = len(os.listdir('./data/raw/sleep-edf'))
        print(f"✅ Sleep-EDF: {sleep_count} files in ./data/raw/sleep-edf/")
    
    if os.path.exists('./data/raw/wesad'):
        wesad_count = len(os.listdir('./data/raw/wesad'))
        print(f"✅ WESAD: {wesad_count} files in ./data/raw/wesad/")
    
    print("\n🎉 Data organization complete!")
    print("You can now run the preprocessing notebooks.")

if __name__ == "__main__":
    main()
