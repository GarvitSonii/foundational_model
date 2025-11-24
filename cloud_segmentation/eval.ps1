Write-Host "`nBegining!"
python evaluate_snn.py `
    --checkpoint "C:\Users\garvi\Desktop\saver\saved_work\model_3.pt" `
    --split_file "train_val_split.pkl" `
    --batch_size 16 `
    --num_workers 0 `
    --device "cpu" `
    --timesteps 6

Write-Host "`nDone!"