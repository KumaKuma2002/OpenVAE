#!/bin/bash
export INPUT_ROOT=/projects/bodymaps/jliu452/Data/Dataset901_SMILE/PT_data
export OUTPUT_ROOT=/projects/bodymaps/jliu452/Data/Dataset804_SMILE-SR_Validation

python validation_degrader.py \
    --patient_id "baichaoxiao20240416_arterial" \
    --input_root "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}"

python validation_degrader.py \
    --patient_id "baichaoxiao20240416_venous" \
    --input_root "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}"

python validation_degrader.py \
    --patient_id "RS-GIST-121_venous" \
    --input_root "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}"

python validation_degrader.py \
    --patient_id "WAW-TACE333_venous" \
    --input_root "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}"