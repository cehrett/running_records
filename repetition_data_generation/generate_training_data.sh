source ../venv/bin/activate
echo "Generating Text Data..."
python create_text_training_data.py --input_file data/unprocessedSentences.txt --output_file data/output --max_length $1
echo "Generating Audio Data for Repetitions"
python create_audio_training_data.py --input_file data/output_repetitions.csv --output_file data/rep_audio
echo "Generating Audio Data for Deletitons"
python create_audio_training_data.py --input_file data/output_deletions.csv --output_file data/del_audio
#echo "Generating Audio Data for Substitutions"
#python create_audio_training_data.py --input_file data/output_substitutions.csv --output_file data/sub_audio
echo "Deleting Audio Files"
rm data/rep_audio_* data/sub_audio_* data/del_audio_* -f
git add data
git commit -m "Automated Training Run"
git push
echo "DONE!"