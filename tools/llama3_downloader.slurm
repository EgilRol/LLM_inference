#!/bin/bash
#SBATCH --job-name=llama3-download
#SBATCH --output=llama3_downloader_%j.out
#SBATCH --error=llama3_downloader_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --partition=general

uv run python llama3_downloader.py --out /shared/home/egr776/project/LLM_inference/assets
