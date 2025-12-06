# F0 Estimation Evaluation

This evaluation script compares 7 different F0 estimation methods on the vocadito dataset:

## Methods Evaluated

1. **AutoCorrelation** - Baseline autocorrelation method
2. **SimpleNet_BCE** - Simple CNN with BCE loss
3. **SimpleNet_BCE+MSE** - Simple CNN with hybrid BCE+MSE loss
4. **ResNet_BCE_Frozen** - ResNet-18 with BCE loss (frozen backbone)
5. **ResNet_BCE_Unfrozen** - ResNet-18 with BCE loss (unfrozen backbone)
6. **ResNet_BCE+MSE_Frozen** - ResNet-18 with hybrid loss (frozen backbone)
7. **ResNet_BCE+MSE_Unfrozen** - ResNet-18 with hybrid loss (unfrozen backbone)

## Setup on PACE

### 1. Load required modules
```bash
module load anaconda3
module load cuda/11.7  # or appropriate CUDA version
```

### 2. Create/activate conda environment
```bash
conda create -n f0_eval python=3.9
conda activate f0_eval
```

### 3. Install dependencies
```bash
pip install -r requirements_eval.txt
```

### 4. Run evaluation
```bash
# Interactive job (for testing)
salloc --gres=gpu:1 --mem=16G -N1 -n4 -t 2:00:00
python evaluate.py

# OR submit as batch job (recommended)
sbatch run_evaluation.pbs
```

## Output

The script will:
- Evaluate all 7 methods on all vocadito test files
- Print results for each file and method
- Calculate average metrics across all files
- Save results to `evaluation_results.csv`

## Metrics

The script uses `mir_eval.melody` to compute:
- **Voicing Recall** - Percentage of voiced frames correctly identified
- **Voicing False Alarm** - Percentage of unvoiced frames incorrectly identified as voiced
- **Raw Pitch Accuracy** - Percentage of voiced frames with correct pitch (within 50 cents)
- **Raw Chroma Accuracy** - Percentage of voiced frames with correct chroma
- **Overall Accuracy** - Combined voicing and pitch accuracy

## Notes

- The autocorrelation baseline uses frame_size=2048, hop_ratio=0.5
- Deep learning models use HCQT with 6 harmonics [0.5, 1, 2, 3, 4, 5]
- F0 range: 50-800 Hz for autocorrelation, full CQT range for DL models
- Salience threshold: 0.5 for DL models, 0.25 for autocorrelation
