#!/bin/bash
#SBATCH --job-name=dumper
#SBATCH --output=dumper_%j.out
#SBATCH --error=dumper_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=16
#SBATCH --partition=seas_compute

uv run python dumper.py