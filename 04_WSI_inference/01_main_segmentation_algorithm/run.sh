#!/bin/bash
# setting
SLIDE_FOLDER="/data/source/WSI_dataset1"
OUTPUT_DIR="/data/output/WSI_dataset1"

DEVICES=('MIG-eaeb7a8c-8788-5dfb-a860-ad63442bebd0'
	'MIG-fb02ca88-7d0c-5b31-9692-4962e5817356')

NUM_PROCESSES=${#DEVICES[@]}

if [ "$NUM_PROCESSES" -eq 1 ]
then
  # run wsi_tis_detect.py
  python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"
else
  CUDA_VISIBLE_DEVICES=${DEVICES[0]} python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"
fi

# calculate the number of WSIs
TOTAL_FILES=$(ls -1 "$SLIDE_FOLDER" | wc -l)

# WSIs for each process
FILES_PER_PROCESS=$(( (TOTAL_FILES + NUM_PROCESSES - 1) / NUM_PROCESSES ))

# run main.py
for ((i=1; i<=NUM_PROCESSES; i++)); do
    cp main.py main"$i".py
    MIG_CODE=${DEVICES[i-1]}
    START=$(( (i - 1) * FILES_PER_PROCESS ))
    END=$(( i * FILES_PER_PROCESS ))
    if [ "$NUM_PROCESSES" -eq 1 ]
    then
      python main"$i".py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR" --start "$START" --end "$END"
    else
      CUDA_VISIBLE_DEVICES="${MIG_CODE}" python main"$i".py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR" --start "$START" --end "$END" &
    fi
done

wait

echo "All processes completed!"
