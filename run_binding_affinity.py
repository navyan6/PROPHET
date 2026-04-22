from pathlib import Path
from PeptiVerse.inference import PeptiVersePredictor

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    predictor = PeptiVersePredictor(
        manifest_path=root / "PeptiVerse" / "best_models.txt",
        classifier_weight_root=root / "PeptiVerse",
        only_properties=["binding_affinity"]
    )

    seq = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    binder = "GIGAVLKVLT"
    result = predictor.predict_binding_affinity("wt", target_seq=seq, binder_str=binder)
    print("Binding affinity result:", result)

    # Example with uncertainty
    result_unc = predictor.predict_binding_affinity("wt", target_seq=seq, binder_str=binder, uncertainty=True)
    print("Binding affinity with uncertainty:", result_unc)
