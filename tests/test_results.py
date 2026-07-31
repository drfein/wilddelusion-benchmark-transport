from pathlib import Path

from wilddelusion_repro.integrity import verify_packaged_results


def test_packaged_claims_pass():
    root = Path(__file__).resolve().parents[1]
    report = verify_packaged_results(root)
    assert report["status"] == "pass"
    assert report["selected_claims"] == 5
    assert len(report["evidence_families"]) == 5
