#!/bin/bash
set -euo pipefail

cd /path/to/PhysicalAI_Track1

SLICE_JOB=$(qsub \
  -W depend=afterok:710362.bright04 \
  -v SPLIT=train,PCD_ROOT=/path/to/scratch/PhysicalAI_Track1/depth_estimates/zio_pcd_depthpro_train_stride30,OUT_DIR=/path/to/scratch/PhysicalAI_Track1/depth_estimates/zio_pcd_depthpro_train_stride30_sliced \
  scripts/pbs/repair_depthpro_pcd_gt_and_slice.pbs)
echo "SLICE_JOB=${SLICE_JOB}"

BALANCE_JOB=$(qsub \
  -W depend=afterok:${SLICE_JOB} \
  -v SRC_ROOT=/path/to/scratch/PhysicalAI_Track1/depth_estimates/zio_pcd_depthpro_trainval_stride30_sliced,TRAIN_SLICED_ROOT=/path/to/scratch/PhysicalAI_Track1/depth_estimates/zio_pcd_depthpro_train_stride30_sliced,VAL_SLICED_ROOT=/path/to/scratch/PhysicalAI_Track1/depth_estimates/zio_pcd_depthpro_val_stride30_sliced,OUT_ROOT=/path/to/scratch/PhysicalAI_Track1/depth_estimates/zio_pcd_depthpro_trainval_stride30_balanced60k,MAX_TRAIN=60000,MAX_PERSON_ONLY=12000 \
  scripts/pbs/build_depthpro_balanced_subset_after_train_slice.pbs)
echo "BALANCE_JOB=${BALANCE_JOB}"
