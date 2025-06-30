

Submit to TIRA via:

```
tira-cli code-submission \
    --mount-hf-model sentence-transformers/LaBSE alimrn001/pan2025-multi-author-models \
	--path . --task multi-author-writing-style-analysis-2025 \
	--dataset multi-author-writing-spot-check-20250503-training \
	--command 'python3 /app/predict.py --dataset $inputDataset --output $outputDir'
```
