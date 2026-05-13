#!/bin/bash
#SBATCH --job-name=ArchaIA_ComputeCls
#SBATCH --partition=voltagepark
#SBATCH --account=daphneippolito
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs_cls_%j.out

# 1. Cargar tu entorno de Conda
source ~/.bashrc
conda activate archaia

# 2. Definir la ruta base
BASE_PATH="/home/nicolascs/archaia_image-classification/images"

# 3. Ejecutar el script para cada carpeta (del 00000 al 00038)
# Nota: Usamos {00000..00038} para generar la secuencia con ceros a la izquierda
for folder in {00000..00038}; do
    echo "Procesando carpeta: $folder"
    python ComputeCls.py "$BASE_PATH/$folder"
done