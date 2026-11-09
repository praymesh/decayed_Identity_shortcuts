Task 1 final report, 9 November 2026

Task_1_Report.docx is the consolidated report. An identical copy is saved at
/home/jovyan/Task_1_Report.docx. The report uses the final full-data MAE probes,
not the earlier 1000-image subset scores in reports/task1_200ep/.

mae_results.csv: five matched 200-epoch MAE runs and final full-data metrics.
depth_metrics.csv: rank, frequency energy and spatial variation by MAE block.
sit_results.csv: three SiT runs and the shared-setting feature audit.
class_mapping.csv: exact ImageNet-100 class IDs, labels and split counts.
commands.txt: saved launch commands for training and evaluation.
audit.json: source paths, hashes, checks and the precision-label correction.
figures/: selected report figures generated from recorded measurements.
report_text.txt: searchable text of the report.

Best observed MAE result: Gaussian beta=0.5, 75.66% linear probe and 65.98% KNN.
The report also records the negative beta=1 result and the limitations of SiT.
All results use one training seed. Actual RTX 3070 execution remains unverified.

Historical run logs are unchanged. Some standalone evaluation metadata called
feature extraction BF16, although the executed default CUDA autocast used FP16.
This is documented in audit.json and the report; recorded scores are unchanged.

Large data, checkpoints, caches and raw logs remain under the ignored datasets/,
checkpoints/ and runs/ directories. Back them up separately from Git.
