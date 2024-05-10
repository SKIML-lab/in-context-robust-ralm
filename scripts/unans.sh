#!/bin/bash
#SBATCH -J unans-test
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:0
#SBATCH --mem=200G
#SBATCH -o log/unans.out
#SBATCH -e log/unans.err
#SBATCH --time 48:00:00

export PYTHONPATH=../
python3 ralm/experiment.py \
    --dataset Atipico1/incontext_nq \
    --task unans \
    --demons zeroshot \
    --add_qa_prompt False \
    --qa_demon_size 0 \
    --unans_demon_size 0 \
    --filter_uncertain True

python3 ralm/experiment.py \
    --dataset Atipico1/incontext_nq \
    --task unans \
    --demons ours \
    --add_qa_prompt True \
    --qa_demon_size 1 \
    --unans_demon_size 1 \
    --filter_uncertain True

# python3 ralm/experiment.py \
#     --dataset Atipico1/incontext_nq \
#     --task unans \
#     --demons ours \
#     --add_qa_prompt True \
#     --qa_demon_size 1 \
#     --unans_demon_size 1 \
#     --filter_uncertain True

# python3 ralm/experiment.py \
#     --dataset Atipico1/incontext_nq \
#     --task unans \
#     --demons ours \
#     --add_qa_prompt True \
#     --qa_demon_size 1 \
#     --unans_demon_size 1 \
#     --test False \
#     --filter_uncertain True

# python3 ralm/experiment.py \
#     --dataset Atipico1/incontext_nq \
#     --task unans \
#     --demons zeroshot \
#     --add_qa_prompt True \
#     --qa_demon_size 1 \
#     --unans_demon_size 1 \
#     --test False \
#     --filter_uncertain True