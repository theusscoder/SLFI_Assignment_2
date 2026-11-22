# Reproduce every experiment in Results/ (Windows PowerShell; run from assignment_2b/).
# Equivalent bash commands are listed in README.md.
$ErrorActionPreference = "Stop"
python -m models.common.features
foreach ($mode in "train", "test") {
    python -m "models.speech_pipeline.$mode" --speech_feat wavlm
    python -m "models.speech_pipeline.$mode" --speech_feat mel
    python -m "models.text_pipeline.$mode"
    python -m "models.fusion_pipeline.$mode" --fusion gated
    python -m "models.fusion_pipeline.$mode" --fusion concat
}
python Results/make_results.py
python Results/extra_experiments.py
