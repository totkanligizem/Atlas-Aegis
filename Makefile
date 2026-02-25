.PHONY: env-check lint-data budget-demo smoke tier-fast tier-candidate tier-full track-risk-dry track-risk track-risk-visdrone-mot-val track-risk-visdrone-mot-val-seq2 mot-sweep-visdrone-val mot-sweep-visdrone-val-seq2 compare-mot-sweeps mot-postfilter-sweep mot-postfilter-sweep-recall mot-postfilter-sweep-classmap mot-eval-profile-recall mot-eval-profile-stability mot-profile-compare mot-profile-release-gate mot-release-run pre-release-check pre-release-check-strict release release-strict mot-error-slices build-gt-events-visdrone-val build-gt-events-visdrone-val-seq2 eval-leadtime-visdrone-val eval-leadtime-visdrone-val-seq2 bench-fast-dry bench-candidate-dry bench-full-dry latency-bench-dry latency-bench latency-bench-gate export-bench-dry export-bench export-bench-gate dataset-status data-download-uavdt data-prepare-manual prepare-visdrone-det prepare-visdrone-det-smoke prepare-visdrone-det-corruptaware prepare-visdrone-det-corruptaware-quick prepare-visdrone-det-poisson-focus-quick prepare-visdrone-det-poisson-focus-mid prepare-visdrone-det-poisson-focus-1600 prepare-visdrone-det-blur-rescue weights-yolov8n weights-yolov8s train-det-fast-dry train-det-fast-smoke train-det-fast-full-ep1 train-det-fast-full-ep2-from-ep1 train-det-fast-full-ep3-from-ep2 train-det-highrecall-dry train-det-highrecall-smoke train-det-highrecall-full-ep1 train-det-highrecall-corruptaware-ep1 train-det-highrecall-corruptaware-quick-ep1 train-det-highrecall-poisson-focus-quick-ep1 train-det-highrecall-poisson-focus-mid-ep1 train-det-highrecall-poisson-focus-1600-ep1 train-det-highrecall-blur-rescue-ep1 eval-det-dry eval-det eval-det-smoke-trained eval-det-fast-full-ep1 eval-det-fast-full-ep2 eval-det-fast-full-ep3 eval-det-candidate-fast-full-ep1 eval-det-candidate-fast-full-ep2 eval-det-candidate-fast-full-ep3 eval-det-highrecall-full-ep1 eval-det-candidate-highrecall-full-ep1 eval-det-highrecall-corruptaware-fast eval-det-candidate-highrecall-corruptaware eval-det-highrecall-corruptaware-quick-fast eval-det-candidate-highrecall-corruptaware-quick eval-det-candidate-highrecall-poisson-focus-quick eval-det-candidate-highrecall-poisson-focus-mid eval-det-candidate-highrecall-poisson-focus-1600 eval-det-candidate-poisson-focus-1600-sweep eval-det-candidate-highrecall-blur-rescue eval-det-full41-fast-ep3 eval-det-full41-highrecall-ep1 eval-det-full41-highrecall-corruptaware eval-det-full41-highrecall-poisson-focus-mid eval-det-full41-highrecall-poisson-focus-1600 eval-det-full41-highrecall-blur-rescue eval-det-full41-fast-ep3-bg eval-det-full41-highrecall-ep1-bg mot-eval-dry mot-eval build-mot-visdrone-val build-mot-visdrone-val-seq2 mot-eval-visdrone-val mot-eval-visdrone-val-seq2 dashboard loop-poisson-focus-1600
PYTHON ?= .venv/bin/python
MOT_PROFILE ?= recall
ifeq ($(MOT_PROFILE),recall)
MOT_MIN_TRACK_AGE ?= 6
MOT_MIN_CONF ?= 0.30
MOT_MIN_CONF_RELAXED ?= -1
MOT_MIN_CONF_RELAX_AGE_START ?= 0
MOT_MIN_ROI_DWELL ?= 0
MOT_CLASS_MIN_CONF_MAP ?= 1:0.45,4:0.34,9:0.25
else ifeq ($(MOT_PROFILE),stability)
MOT_MIN_TRACK_AGE ?= 6
MOT_MIN_CONF ?= 0.30
MOT_MIN_CONF_RELAXED ?= -1
MOT_MIN_CONF_RELAX_AGE_START ?= 0
MOT_MIN_ROI_DWELL ?= 0
MOT_CLASS_MIN_CONF_MAP ?= 1:0.45,4:0.34,9:0.30
else ifeq ($(MOT_PROFILE),balanced)
MOT_MIN_TRACK_AGE ?= 8
MOT_MIN_CONF ?= 0.30
MOT_MIN_CONF_RELAXED ?= -1
MOT_MIN_CONF_RELAX_AGE_START ?= 0
MOT_MIN_ROI_DWELL ?= 0
MOT_CLASS_MIN_CONF_MAP ?=
else
$(error Unsupported MOT_PROFILE='$(MOT_PROFILE)'. Use recall|stability|balanced)
endif
MOT_SWEEP_AGE_VALUES ?= 3,4,5,6,8,10
MOT_SWEEP_CONF_VALUES ?= 0.25,0.27,0.30
MOT_SWEEP_ROI_VALUES ?= 0,1,2
MOT_SWEEP_CONF_RELAXED_VALUES ?= -1,0.28
MOT_SWEEP_CONF_RELAX_AGE_VALUES ?= 0,10
MOT_SWEEP_CLASS_MIN_CONF_GRID ?= 1:0.30|0.35|0.40|0.45,0:0.30|0.34|0.38,9:0.25|0.27|0.30|0.33,4:0.30|0.34
export XDG_CACHE_HOME ?= $(CURDIR)/.cache
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib

env-check:
	$(PYTHON) scripts/00_env_check.py

lint-data:
	@if [ -d data/processed/visdrone_det/labels/train ]; then \
		$(PYTHON) scripts/01_data_lint.py --labels-dir data/processed/visdrone_det/labels/train; \
	else \
		$(PYTHON) scripts/01_data_lint.py --labels-dir data/labels/train; \
	fi

budget-demo:
	$(PYTHON) scripts/02_budget_guard_demo.py --config configs/budget.yaml

smoke:
	$(PYTHON) scripts/03_smoke_run.py --config configs/risk.yaml --out reports/smoke_run.json

tier-fast:
	$(PYTHON) scripts/04_eval_tier.py --config configs/pipeline.yaml --tier fast

tier-candidate:
	$(PYTHON) scripts/04_eval_tier.py --config configs/pipeline.yaml --tier candidate

tier-full:
	$(PYTHON) scripts/04_eval_tier.py --config configs/pipeline.yaml --tier full

track-risk-dry:
	$(PYTHON) scripts/05_run_track_risk.py --config configs/inference.yaml --dry-run --frames 120

track-risk:
	$(PYTHON) scripts/05_run_track_risk.py --config configs/inference.yaml

track-risk-visdrone-mot-val:
	$(PYTHON) scripts/05_run_track_risk.py --config configs/inference_visdrone_mot_val.yaml

track-risk-visdrone-mot-val-seq2:
	$(PYTHON) scripts/05_run_track_risk.py --config configs/inference_visdrone_mot_val_seq2.yaml

mot-sweep-visdrone-val:
	$(PYTHON) scripts/18_mot_tuning_sweep_visdrone_val.py --inference-config configs/inference_visdrone_mot_val.yaml --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt --conf-values 0.10,0.15,0.20 --iou-values 0.60,0.70 --ttl-values 45,60 --max-frames 300 --min-track-age $(MOT_MIN_TRACK_AGE) --min-conf $(MOT_MIN_CONF) --min-roi-dwell $(MOT_MIN_ROI_DWELL) --report-out reports/mot_tuning_sweep_visdrone_val_report.json

mot-sweep-visdrone-val-seq2:
	$(PYTHON) scripts/18_mot_tuning_sweep_visdrone_val.py --inference-config configs/inference_visdrone_mot_val_seq2.yaml --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000182_00000_v.txt --conf-values 0.10,0.15,0.20 --iou-values 0.60,0.70 --ttl-values 45,60 --max-frames 300 --min-track-age $(MOT_MIN_TRACK_AGE) --min-conf $(MOT_MIN_CONF) --min-roi-dwell $(MOT_MIN_ROI_DWELL) --out-dir logs/mot_sweep_visdrone_val_seq2 --report-out reports/mot_tuning_sweep_visdrone_val_seq2_report.json

compare-mot-sweeps:
	$(PYTHON) scripts/19_compare_mot_sweeps.py --seq1 reports/mot_tuning_sweep_visdrone_val_report.json --seq2 reports/mot_tuning_sweep_visdrone_val_seq2_report.json --out-json reports/mot_sweep_generalization_comparison_report.json --out-md reports/mot_sweep_generalization_comparison_report.md

mot-postfilter-sweep:
	$(PYTHON) scripts/22_mot_postfilter_sweep.py --seq1-gt data/mot/gt.txt --seq1-track-jsonl logs/track_risk_visdrone_mot_val.jsonl --seq2-gt data/mot/gt_seq2.txt --seq2-track-jsonl logs/track_risk_visdrone_mot_val_seq2.jsonl --max-frames 300 --age-values "$(MOT_SWEEP_AGE_VALUES)" --conf-values "$(MOT_SWEEP_CONF_VALUES)" --class-min-conf-map "$(MOT_CLASS_MIN_CONF_MAP)" --conf-relaxed-values="$(MOT_SWEEP_CONF_RELAXED_VALUES)" --conf-relax-age-values "$(MOT_SWEEP_CONF_RELAX_AGE_VALUES)" --roi-values "$(MOT_SWEEP_ROI_VALUES)" --min-seq2-recall 0.30 --min-seq1-recall 0.42 --out-json reports/mot_postfilter_sweep_report.json --out-md reports/mot_postfilter_sweep_report.md

mot-postfilter-sweep-recall:
	$(PYTHON) scripts/22_mot_postfilter_sweep.py --seq1-gt data/mot/gt.txt --seq1-track-jsonl logs/track_risk_visdrone_mot_val.jsonl --seq2-gt data/mot/gt_seq2.txt --seq2-track-jsonl logs/track_risk_visdrone_mot_val_seq2.jsonl --max-frames 300 --age-values "$(MOT_SWEEP_AGE_VALUES)" --conf-values "$(MOT_SWEEP_CONF_VALUES)" --class-min-conf-map "$(MOT_CLASS_MIN_CONF_MAP)" --conf-relaxed-values="$(MOT_SWEEP_CONF_RELAXED_VALUES)" --conf-relax-age-values "$(MOT_SWEEP_CONF_RELAX_AGE_VALUES)" --roi-values "$(MOT_SWEEP_ROI_VALUES)" --min-seq2-recall 0.32 --min-seq1-recall 0.44 --out-json reports/mot_postfilter_sweep_recall_report.json --out-md reports/mot_postfilter_sweep_recall_report.md

mot-postfilter-sweep-classmap:
	$(PYTHON) scripts/22_mot_postfilter_sweep.py --seq1-gt data/mot/gt.txt --seq1-track-jsonl logs/track_risk_visdrone_mot_val.jsonl --seq2-gt data/mot/gt_seq2.txt --seq2-track-jsonl logs/track_risk_visdrone_mot_val_seq2.jsonl --max-frames 300 --age-values "6,8" --conf-values "0.30" --class-min-conf-map "" --class-min-conf-grid "$(MOT_SWEEP_CLASS_MIN_CONF_GRID)" --conf-relaxed-values="-1" --conf-relax-age-values "0" --roi-values "0" --min-seq2-recall 0.333 --min-seq1-recall 0.44 --out-json reports/mot_postfilter_sweep_classmap_report.json --out-md reports/mot_postfilter_sweep_classmap_report.md

mot-eval-profile-recall:
	$(PYTHON) scripts/17_build_mot_eval_files.py --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt --pred-jsonl logs/track_risk_visdrone_mot_val.jsonl --max-frames 300 --min-track-age 6 --min-conf 0.30 --class-min-conf-map "1:0.45,4:0.34,9:0.25" --min-conf-relaxed -1 --min-conf-relax-age-start 0 --min-roi-dwell 0 --out-gt data/mot/gt_profile_recall_seq1.txt --out-pred data/mot/pred_profile_recall_seq1.txt --report-out reports/mot_build_profile_recall_seq1_report.json
	$(PYTHON) scripts/17_build_mot_eval_files.py --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000182_00000_v.txt --pred-jsonl logs/track_risk_visdrone_mot_val_seq2.jsonl --max-frames 300 --min-track-age 6 --min-conf 0.30 --class-min-conf-map "1:0.45,4:0.34,9:0.25" --min-conf-relaxed -1 --min-conf-relax-age-start 0 --min-roi-dwell 0 --out-gt data/mot/gt_profile_recall_seq2.txt --out-pred data/mot/pred_profile_recall_seq2.txt --report-out reports/mot_build_profile_recall_seq2_report.json
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt_profile_recall_seq1.txt --pred-mot data/mot/pred_profile_recall_seq1.txt --pred-events reports/track_risk_visdrone_mot_val_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_profile_recall_seq1_report.json
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt_profile_recall_seq2.txt --pred-mot data/mot/pred_profile_recall_seq2.txt --pred-events reports/track_risk_visdrone_mot_val_seq2_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_profile_recall_seq2_report.json

mot-eval-profile-stability:
	$(PYTHON) scripts/17_build_mot_eval_files.py --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt --pred-jsonl logs/track_risk_visdrone_mot_val.jsonl --max-frames 300 --min-track-age 6 --min-conf 0.30 --class-min-conf-map "1:0.45,4:0.34,9:0.30" --min-conf-relaxed -1 --min-conf-relax-age-start 0 --min-roi-dwell 0 --out-gt data/mot/gt_profile_stability_seq1.txt --out-pred data/mot/pred_profile_stability_seq1.txt --report-out reports/mot_build_profile_stability_seq1_report.json
	$(PYTHON) scripts/17_build_mot_eval_files.py --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000182_00000_v.txt --pred-jsonl logs/track_risk_visdrone_mot_val_seq2.jsonl --max-frames 300 --min-track-age 6 --min-conf 0.30 --class-min-conf-map "1:0.45,4:0.34,9:0.30" --min-conf-relaxed -1 --min-conf-relax-age-start 0 --min-roi-dwell 0 --out-gt data/mot/gt_profile_stability_seq2.txt --out-pred data/mot/pred_profile_stability_seq2.txt --report-out reports/mot_build_profile_stability_seq2_report.json
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt_profile_stability_seq1.txt --pred-mot data/mot/pred_profile_stability_seq1.txt --pred-events reports/track_risk_visdrone_mot_val_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_profile_stability_seq1_report.json
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt_profile_stability_seq2.txt --pred-mot data/mot/pred_profile_stability_seq2.txt --pred-events reports/track_risk_visdrone_mot_val_seq2_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_profile_stability_seq2_report.json

mot-profile-compare: mot-eval-profile-recall mot-eval-profile-stability
	$(PYTHON) scripts/24_compare_mot_profiles.py --recall-seq1 reports/mot_eval_profile_recall_seq1_report.json --recall-seq2 reports/mot_eval_profile_recall_seq2_report.json --stability-seq1 reports/mot_eval_profile_stability_seq1_report.json --stability-seq2 reports/mot_eval_profile_stability_seq2_report.json --out-json reports/mot_profile_comparison_report.json --out-md reports/mot_profile_comparison_report.md

mot-profile-release-gate: mot-profile-compare
	$(PYTHON) scripts/25_select_mot_release_profile.py --comparison reports/mot_profile_comparison_report.json --objective hota --min-seq2-recall 0.333 --min-seq2-hota 0.300 --min-seq2-mota -0.080 --fail-on-no-go --out-json reports/mot_profile_release_gate_report.json --out-md reports/mot_profile_release_gate_report.md --out-env reports/mot_profile_release.env

mot-release-run: mot-profile-release-gate
	@set -a; . reports/mot_profile_release.env; set +a; \
	if [ "$$MOT_RELEASE_GATE_STATUS" != "PASS" ]; then \
		echo "[mot-release] gate is $$MOT_RELEASE_GATE_STATUS, stopping"; \
		exit 2; \
	fi; \
	echo "[mot-release] using MOT_PROFILE=$$MOT_PROFILE"; \
	MOT_PROFILE=$$MOT_PROFILE $(MAKE) build-mot-visdrone-val build-mot-visdrone-val-seq2 mot-eval-visdrone-val mot-eval-visdrone-val-seq2

pre-release-check:
	$(PYTHON) scripts/26_pre_release_check.py

pre-release-check-strict:
	$(PYTHON) scripts/26_pre_release_check.py --strict

release: pre-release-check
	@echo "[release] PASS: pre-release check passed and release artifacts are ready."

release-strict: pre-release-check-strict
	@echo "[release-strict] PASS: strict pre-release check passed."

mot-error-slices:
	$(PYTHON) scripts/21_mot_error_slices.py --seq1-gt data/mot/gt.txt --seq1-pred data/mot/pred.txt --seq2-gt data/mot/gt_seq2.txt --seq2-pred data/mot/pred_seq2.txt --out-json reports/mot_error_slices_report.json --out-md reports/mot_error_slices_report.md

build-gt-events-visdrone-val:
	$(PYTHON) scripts/20_build_gt_events_from_visdrone_mot.py --annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt --sequence-dir data/manual/visdrone/VisDrone2019-MOT-val/sequences/uav0000086_00000_v --roi-x1 0.20 --roi-y1 0.20 --roi-x2 0.80 --roi-y2 0.80 --min-frames-in-roi 10 --max-frames 300 --out-events data/mot/gt_events_visdrone_val.json --report-out reports/gt_event_build_visdrone_val_report.json

build-gt-events-visdrone-val-seq2:
	$(PYTHON) scripts/20_build_gt_events_from_visdrone_mot.py --annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000182_00000_v.txt --sequence-dir data/manual/visdrone/VisDrone2019-MOT-val/sequences/uav0000182_00000_v --roi-x1 0.20 --roi-y1 0.20 --roi-x2 0.80 --roi-y2 0.80 --min-frames-in-roi 10 --max-frames 300 --out-events data/mot/gt_events_visdrone_val_seq2.json --report-out reports/gt_event_build_visdrone_val_seq2_report.json

eval-leadtime-visdrone-val:
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt.txt --pred-mot data/mot/pred.txt --gt-events data/mot/gt_events_visdrone_val.json --require-gt-events --pred-events reports/track_risk_visdrone_mot_val_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_visdrone_val_with_events_report.json

eval-leadtime-visdrone-val-seq2:
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt_seq2.txt --pred-mot data/mot/pred_seq2.txt --gt-events data/mot/gt_events_visdrone_val_seq2.json --require-gt-events --pred-events reports/track_risk_visdrone_mot_val_seq2_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_visdrone_val_seq2_with_events_report.json

bench-fast-dry:
	$(PYTHON) scripts/06_benchmark_corruptions.py --config configs/benchmark.yaml --pipeline-config configs/pipeline.yaml --tier fast --dry-run --output-report reports/benchmark_fast_dry_run_report.json

bench-candidate-dry:
	$(PYTHON) scripts/06_benchmark_corruptions.py --config configs/benchmark.yaml --pipeline-config configs/pipeline.yaml --tier candidate --dry-run --output-report reports/benchmark_candidate_dry_run_report.json

bench-full-dry:
	$(PYTHON) scripts/06_benchmark_corruptions.py --config configs/benchmark.yaml --pipeline-config configs/pipeline.yaml --tier full --dry-run --output-report reports/benchmark_full_dry_run_report.json

latency-bench-dry:
	$(PYTHON) scripts/27_benchmark_latency.py --config configs/latency_benchmark.yaml --dry-run --out-report reports/latency_benchmark_dry_run_report.json --out-md reports/latency_benchmark_dry_run_report.md

latency-bench:
	$(PYTHON) scripts/27_benchmark_latency.py --config configs/latency_benchmark.yaml

latency-bench-gate:
	$(PYTHON) scripts/27_benchmark_latency.py --config configs/latency_benchmark.yaml --fail-on-gate

export-bench-dry:
	$(PYTHON) scripts/28_benchmark_exports.py --config configs/export_benchmark.yaml --dry-run --out-report reports/export_benchmark_dry_run_report.json --out-md reports/export_benchmark_dry_run_report.md

export-bench:
	$(PYTHON) scripts/28_benchmark_exports.py --config configs/export_benchmark.yaml

export-bench-gate:
	$(PYTHON) scripts/28_benchmark_exports.py --config configs/export_benchmark.yaml --fail-on-gate

dataset-status:
	$(PYTHON) scripts/07_dataset_status.py

data-download-uavdt:
	bash scripts/01_download_data.sh uavdt

data-prepare-manual:
	bash scripts/01_prepare_manual_data.sh

prepare-visdrone-det:
	$(PYTHON) scripts/08_prepare_visdrone_det.py --raw-root data/raw/visdrone --out-root data/processed/visdrone_det --force

prepare-visdrone-det-smoke:
	$(PYTHON) scripts/08_prepare_visdrone_det.py --raw-root data/raw/visdrone --out-root data/processed/visdrone_det_smoke --force --max-train-images 512 --max-val-images 120 --report-out reports/visdrone_prepare_smoke_report.json

prepare-visdrone-det-corruptaware:
	$(PYTHON) scripts/12_build_corruption_aug_det.py --src-root data/processed/visdrone_det --out-root data/processed/visdrone_det_corruptaware --conditions s3_blur,s3_poisson --max-train-images 2000 --force --report-out reports/visdrone_corruptaware_prepare_report.json

prepare-visdrone-det-corruptaware-quick:
	$(PYTHON) scripts/12_build_corruption_aug_det.py --src-root data/processed/visdrone_det --out-root data/processed/visdrone_det_corruptaware_quick --conditions s3_blur,s3_poisson --max-train-images 600 --force --report-out reports/visdrone_corruptaware_quick_prepare_report.json

prepare-visdrone-det-poisson-focus-quick:
	$(PYTHON) scripts/12_build_corruption_aug_det.py --src-root data/processed/visdrone_det --out-root data/processed/visdrone_det_poisson_focus_quick --conditions s3_poisson,poisson_s4,s3_blur --max-train-images 600 --force --report-out reports/visdrone_poisson_focus_quick_prepare_report.json

prepare-visdrone-det-poisson-focus-mid:
	$(PYTHON) scripts/12_build_corruption_aug_det.py --src-root data/processed/visdrone_det --out-root data/processed/visdrone_det_poisson_focus_mid --conditions s3_poisson,poisson_s4,s3_blur --max-train-images 1200 --force --report-out reports/visdrone_poisson_focus_mid_prepare_report.json

prepare-visdrone-det-poisson-focus-1600:
	$(PYTHON) scripts/12_build_corruption_aug_det.py --src-root data/processed/visdrone_det --out-root data/processed/visdrone_det_poisson_focus_1600 --conditions s3_poisson,poisson_s3,poisson_s4,s3_blur --max-train-images 1600 --force --report-out reports/visdrone_poisson_focus_1600_prepare_report.json

prepare-visdrone-det-blur-rescue:
	$(PYTHON) scripts/12_build_corruption_aug_det.py --src-root data/processed/visdrone_det --out-root data/processed/visdrone_det_blur_rescue --conditions s3_blur,blur_s3,s3_poisson,poisson_s3 --max-train-images 1400 --force --report-out reports/visdrone_blur_rescue_prepare_report.json

weights-yolov8n:
	bash scripts/11_download_weights.sh yolov8n

weights-yolov8s:
	bash scripts/11_download_weights.sh yolov8s

train-det-fast-dry:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_fast.yaml --data data/processed/visdrone_det/dataset.yaml --dry-run

train-det-fast-smoke:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_fast.yaml --data data/processed/visdrone_det_smoke/dataset.yaml --epochs 1 --name visdrone_fast_smoke_ep1

train-det-fast-full-ep1:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_fast.yaml --data data/processed/visdrone_det/dataset.yaml --epochs 1 --name visdrone_fast_full_ep1_cpu

train-det-fast-full-ep2-from-ep1:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_fast.yaml --model runs/detect/artifacts/detect/visdrone_fast_full_ep1_cpu/weights/best.pt --data data/processed/visdrone_det/dataset.yaml --epochs 1 --name visdrone_fast_full_ep2_cpu

train-det-fast-full-ep3-from-ep2:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_fast.yaml --model runs/detect/artifacts/detect/visdrone_fast_full_ep2_cpu/weights/best.pt --data data/processed/visdrone_det/dataset.yaml --epochs 1 --name visdrone_fast_full_ep3_cpu

train-det-highrecall-dry:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall.yaml --data data/processed/visdrone_det/dataset.yaml --dry-run

train-det-highrecall-smoke:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall.yaml --data data/processed/visdrone_det_smoke/dataset.yaml --epochs 1 --name visdrone_highrecall_smoke_ep1

train-det-highrecall-full-ep1:
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall.yaml --data data/processed/visdrone_det/dataset.yaml --epochs 1 --name visdrone_highrecall_full_ep1_cpu

train-det-highrecall-corruptaware-ep1:
	@RUN_NAME=$${RUN_NAME:-visdrone_highrecall_corruptaware_$$(date +%Y%m%d_%H%M%S)}; \
	echo "training run_name=$$RUN_NAME"; \
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall_corruptaware.yaml --data data/processed/visdrone_det_corruptaware/dataset.yaml --epochs 1 --name "$$RUN_NAME"

train-det-highrecall-corruptaware-quick-ep1:
	@RUN_NAME=$${RUN_NAME:-visdrone_highrecall_corruptaware_quick_$$(date +%Y%m%d_%H%M%S)}; \
	echo "training run_name=$$RUN_NAME"; \
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall_corruptaware_quick.yaml --data data/processed/visdrone_det_corruptaware_quick/dataset.yaml --epochs 1 --name "$$RUN_NAME" --device cpu

train-det-highrecall-poisson-focus-quick-ep1:
	@RUN_NAME=$${RUN_NAME:-visdrone_highrecall_poisson_focus_quick_$$(date +%Y%m%d_%H%M%S)}; \
	echo "training run_name=$$RUN_NAME"; \
	BASE_MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_corruptaware_report.json); \
	echo "base_model=$$BASE_MODEL"; \
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall_poisson_focus_quick.yaml --data data/processed/visdrone_det_poisson_focus_quick/dataset.yaml --epochs 1 --name "$$RUN_NAME" --model "$$BASE_MODEL" --device cpu

train-det-highrecall-poisson-focus-mid-ep1:
	@RUN_NAME=$${RUN_NAME:-visdrone_highrecall_poisson_focus_mid_$$(date +%Y%m%d_%H%M%S)}; \
	echo "training run_name=$$RUN_NAME"; \
	BASE_MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_corruptaware_report.json); \
	echo "base_model=$$BASE_MODEL"; \
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall_poisson_focus_mid.yaml --data data/processed/visdrone_det_poisson_focus_mid/dataset.yaml --epochs 1 --name "$$RUN_NAME" --model "$$BASE_MODEL" --device cpu

train-det-highrecall-poisson-focus-1600-ep1:
	@RUN_NAME=$${RUN_NAME:-visdrone_highrecall_poisson_focus_1600_$$(date +%Y%m%d_%H%M%S)}; \
	echo "training run_name=$$RUN_NAME"; \
	BASE_MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_corruptaware_report.json); \
	echo "base_model=$$BASE_MODEL"; \
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall_poisson_focus_1600.yaml --data data/processed/visdrone_det_poisson_focus_1600/dataset.yaml --epochs 1 --name "$$RUN_NAME" --model "$$BASE_MODEL" --device cpu

train-det-highrecall-blur-rescue-ep1:
	@RUN_NAME=$${RUN_NAME:-visdrone_highrecall_blur_rescue_$$(date +%Y%m%d_%H%M%S)}; \
	echo "training run_name=$$RUN_NAME"; \
	BASE_MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_poisson_focus_mid_report.json); \
	echo "base_model=$$BASE_MODEL"; \
	$(PYTHON) scripts/09_train_detector.py --config configs/detector_highrecall_blur_rescue.yaml --data data/processed/visdrone_det_blur_rescue/dataset.yaml --epochs 1 --name "$$RUN_NAME" --model "$$BASE_MODEL" --device cpu

eval-det-dry:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval.yaml --dry-run

eval-det:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval.yaml

eval-det-smoke-trained:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval.yaml --model runs/detect/artifacts/detect/visdrone_fast_smoke_ep1/weights/best.pt --data-root data/processed/visdrone_det_smoke --split val --max-images 120 --output-report reports/detector_eval_smoke_trained_report.json

eval-det-fast-full-ep1:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval.yaml --model runs/detect/artifacts/detect/visdrone_fast_full_ep1_cpu/weights/best.pt --output-report reports/detector_eval_full_ep1_report.json

eval-det-fast-full-ep2:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval.yaml --model runs/detect/artifacts/detect/visdrone_fast_full_ep2_cpu/weights/best.pt --output-report reports/detector_eval_full_ep2_report.json

eval-det-fast-full-ep3:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval.yaml --model runs/detect/artifacts/detect/visdrone_fast_full_ep3_cpu/weights/best.pt --output-report reports/detector_eval_full_ep3_report.json

eval-det-candidate-fast-full-ep1:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate.yaml

eval-det-candidate-fast-full-ep2:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate.yaml --model runs/detect/artifacts/detect/visdrone_fast_full_ep2_cpu/weights/best.pt --output-report reports/detector_eval_candidate_ep2_report.json

eval-det-candidate-fast-full-ep3:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate.yaml --model runs/detect/artifacts/detect/visdrone_fast_full_ep3_cpu/weights/best.pt --output-report reports/detector_eval_candidate_ep3_report.json

eval-det-highrecall-full-ep1:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_highrecall.yaml

eval-det-candidate-highrecall-full-ep1:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate_highrecall.yaml

eval-det-highrecall-corruptaware-fast:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_corruptaware_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_highrecall_corruptaware_fast.yaml --model "$$MODEL" --output-report reports/detector_eval_highrecall_corruptaware_full_fast_report.json

eval-det-candidate-highrecall-corruptaware:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_corruptaware_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate_highrecall_corruptaware.yaml --model "$$MODEL" --output-report reports/detector_eval_candidate_highrecall_corruptaware_full_report.json

eval-det-highrecall-corruptaware-quick-fast:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_highrecall_corruptaware_fast.yaml --model runs/detect/artifacts/detect/visdrone_highrecall_corruptaware_quick_ep1/weights/best.pt --output-report reports/detector_eval_highrecall_corruptaware_quick_fast_report.json

eval-det-candidate-highrecall-corruptaware-quick:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate_highrecall_corruptaware.yaml --model runs/detect/artifacts/detect/visdrone_highrecall_corruptaware_quick_ep1/weights/best.pt --output-report reports/detector_eval_candidate_highrecall_corruptaware_quick_report.json

eval-det-candidate-highrecall-poisson-focus-quick:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_poisson_focus_quick_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate_highrecall_corruptaware.yaml --model "$$MODEL" --output-report reports/detector_eval_candidate_highrecall_poisson_focus_quick_report.json

eval-det-candidate-highrecall-poisson-focus-mid:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_poisson_focus_mid_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate_highrecall_corruptaware.yaml --model "$$MODEL" --output-report reports/detector_eval_candidate_highrecall_poisson_focus_mid_report.json

eval-det-candidate-highrecall-poisson-focus-1600:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_poisson_focus_1600_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate_highrecall_corruptaware.yaml --model "$$MODEL" --output-report reports/detector_eval_candidate_highrecall_poisson_focus_1600_report.json

eval-det-candidate-poisson-focus-1600-sweep:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_poisson_focus_1600_report.json); \
	$(PYTHON) scripts/16_eval_candidate_threshold_sweep.py --config configs/detector_eval_candidate_highrecall_corruptaware.yaml --model "$$MODEL" --conf-values 0.0005,0.001,0.0015 --iou-values 0.55,0.6 --output-report reports/detector_eval_candidate_poisson_focus_1600_sweep_report.json

eval-det-candidate-highrecall-blur-rescue:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_blur_rescue_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_candidate_highrecall_corruptaware.yaml --model "$$MODEL" --output-report reports/detector_eval_candidate_highrecall_blur_rescue_report.json

eval-det-full41-fast-ep3:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_fast.yaml

eval-det-full41-highrecall-ep1:
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_highrecall.yaml

eval-det-full41-highrecall-corruptaware:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_corruptaware_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_highrecall.yaml --model "$$MODEL" --output-report reports/detector_eval_full41_highrecall_corruptaware_full_report.json

eval-det-full41-highrecall-poisson-focus-mid:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_poisson_focus_mid_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_highrecall.yaml --model "$$MODEL" --output-report reports/detector_eval_full41_highrecall_poisson_focus_mid_report.json

eval-det-full41-highrecall-poisson-focus-1600:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_poisson_focus_1600_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_highrecall.yaml --model "$$MODEL" --output-report reports/detector_eval_full41_highrecall_poisson_focus_1600_report.json

eval-det-full41-highrecall-blur-rescue:
	MODEL=$$($(PYTHON) scripts/14_resolve_best_model.py --report reports/train_detector_highrecall_blur_rescue_report.json); \
	$(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_highrecall.yaml --model "$$MODEL" --output-report reports/detector_eval_full41_highrecall_blur_rescue_report.json

eval-det-full41-fast-ep3-bg:
	mkdir -p logs
	nohup $(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_fast.yaml > logs/full41_fast_ep3.log 2>&1 & echo $$! > logs/full41_fast_ep3.pid
	@echo "started full41 fast ep3: PID=$$(cat logs/full41_fast_ep3.pid), log=logs/full41_fast_ep3.log"

eval-det-full41-highrecall-ep1-bg:
	mkdir -p logs
	nohup $(PYTHON) scripts/10_eval_detector.py --config configs/detector_eval_full41_highrecall.yaml > logs/full41_highrecall_ep1.log 2>&1 & echo $$! > logs/full41_highrecall_ep1.pid
	@echo "started full41 highrecall ep1: PID=$$(cat logs/full41_highrecall_ep1.pid), log=logs/full41_highrecall_ep1.log"

mot-eval-dry:
	$(PYTHON) scripts/13_eval_mot.py --dry-run --output-report reports/mot_eval_report.json

mot-eval:
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt.txt --pred-mot data/mot/pred.txt --gt-events data/mot/gt_events_visdrone_val.json --require-gt-events --pred-events reports/track_risk_visdrone_mot_val_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_report.json

build-mot-visdrone-val:
	$(PYTHON) scripts/17_build_mot_eval_files.py --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt --pred-jsonl logs/track_risk_visdrone_mot_val.jsonl --max-frames 300 --min-track-age $(MOT_MIN_TRACK_AGE) --min-conf $(MOT_MIN_CONF) --class-min-conf-map "$(MOT_CLASS_MIN_CONF_MAP)" --min-conf-relaxed $(MOT_MIN_CONF_RELAXED) --min-conf-relax-age-start $(MOT_MIN_CONF_RELAX_AGE_START) --min-roi-dwell $(MOT_MIN_ROI_DWELL) --out-gt data/mot/gt.txt --out-pred data/mot/pred.txt --report-out reports/mot_build_visdrone_val_report.json

build-mot-visdrone-val-seq2:
	$(PYTHON) scripts/17_build_mot_eval_files.py --gt-annotation data/manual/visdrone/VisDrone2019-MOT-val/annotations/uav0000182_00000_v.txt --pred-jsonl logs/track_risk_visdrone_mot_val_seq2.jsonl --max-frames 300 --min-track-age $(MOT_MIN_TRACK_AGE) --min-conf $(MOT_MIN_CONF) --class-min-conf-map "$(MOT_CLASS_MIN_CONF_MAP)" --min-conf-relaxed $(MOT_MIN_CONF_RELAXED) --min-conf-relax-age-start $(MOT_MIN_CONF_RELAX_AGE_START) --min-roi-dwell $(MOT_MIN_ROI_DWELL) --out-gt data/mot/gt_seq2.txt --out-pred data/mot/pred_seq2.txt --report-out reports/mot_build_visdrone_val_seq2_report.json

mot-eval-visdrone-val:
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt.txt --pred-mot data/mot/pred.txt --pred-events reports/track_risk_visdrone_mot_val_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_visdrone_val_report.json

mot-eval-visdrone-val-seq2:
	$(PYTHON) scripts/13_eval_mot.py --gt-mot data/mot/gt_seq2.txt --pred-mot data/mot/pred_seq2.txt --pred-events reports/track_risk_visdrone_mot_val_seq2_report.json --dataset-key visdrone2019_mot --output-report reports/mot_eval_visdrone_val_seq2_report.json

dashboard:
	streamlit run app/dashboard.py

loop-poisson-focus-1600:
	bash scripts/15_run_poisson_focus_1600_loop.sh
