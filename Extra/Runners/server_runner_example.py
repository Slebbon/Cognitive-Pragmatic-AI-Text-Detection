from feature_analysis_utils import FeatureAnalyzer

analyzer = FeatureAnalyzer(output_dir="results")

analyzer.add_feature_family(
    name="metacognition",
    csv_path=r".\metacognition_features_final",
)

#LOAD if they have AI LABELS
analyzer.load_all()

# Run full analysis
analyzer.run_full_analysis(include_shap=True)