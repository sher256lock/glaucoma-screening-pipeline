"""Unit tests for label mapping, ungradable exclusion, and split integrity."""

from pathlib import Path
import pytest

from src.data.load_refuge import map_label
from src.data.create_splits import verify_split_disjointness, CONTRACT_COLUMNS
from src.data.validate_data import validate_image_file


class TestLabelMapping:
    """Test suite for binary label mapping rules."""

    def test_glaucoma_mapping_positive(self):
        """Verify Glaucoma and variants map strictly to positive class (1)."""
        assert map_label("Glaucoma") == 1
        assert map_label("glaucoma") == 1
        assert map_label("1") == 1
        assert map_label(1) == 1

    def test_normal_mapping_negative(self):
        """Verify Normal and Non-Glaucoma map strictly to negative class (0)."""
        assert map_label("Normal") == 0
        assert map_label("normal") == 0
        assert map_label("Non-Glaucoma") == 0
        assert map_label("non-glaucoma") == 0
        assert map_label("0") == 0
        assert map_label(0) == 0

    def test_ungradable_and_suspect_exclusion(self):
        """Verify Ungradable and Suspect samples return None (excluded from manifest)."""
        assert map_label("Ungradable") is None
        assert map_label("ungradable") is None
        assert map_label("Suspect") is None
        assert map_label("suspect") is None

    @pytest.mark.parametrize("invalid_label", ["Unknown", "corrupt", "AMD", "DR", "", " "])
    def test_unknown_label_raises_value_error(self, invalid_label):
        """Verify that unrecognized or corrupt labels raise a structured ValueError with zero silent drops."""
        with pytest.raises(ValueError, match="Unrecognized label encountered"):
            map_label(invalid_label)


class TestSplitIntegrity:
    """Test suite for split disjointness and leakage prevention."""

    def test_split_disjointness_clean(self):
        """Verify that disjoint splits pass without error."""
        records = [
            {"image_path": "train/img1.jpg", "split": "train"},
            {"image_path": "train/img2.jpg", "split": "train"},
            {"image_path": "val/img3.jpg", "split": "val"},
            {"image_path": "val/img4.jpg", "split": "val"},
            {"image_path": "test/img5.jpg", "split": "test"},
            {"image_path": "test/img6.jpg", "split": "test"},
        ]
        # Should not raise
        verify_split_disjointness(records)

    def test_split_disjointness_detects_train_val_overlap(self):
        """Verify that overlap between train and val raises ValueError."""
        records = [
            {"image_path": "shared/img1.jpg", "split": "train"},
            {"image_path": "shared/img1.jpg", "split": "val"},
            {"image_path": "test/img2.jpg", "split": "test"},
        ]
        with pytest.raises(ValueError, match="Train and Val splits overlap"):
            verify_split_disjointness(records)

    def test_split_disjointness_detects_train_test_overlap(self):
        """Verify that overlap between train and test raises ValueError."""
        records = [
            {"image_path": "shared/img1.jpg", "split": "train"},
            {"image_path": "val/img2.jpg", "split": "val"},
            {"image_path": "shared/img1.jpg", "split": "test"},
        ]
        with pytest.raises(ValueError, match="Train and Test splits overlap"):
            verify_split_disjointness(records)

    def test_split_disjointness_detects_val_test_overlap(self):
        """Verify that overlap between val and test raises ValueError."""
        records = [
            {"image_path": "train/img1.jpg", "split": "train"},
            {"image_path": "shared/img2.jpg", "split": "val"},
            {"image_path": "shared/img2.jpg", "split": "test"},
        ]
        with pytest.raises(ValueError, match="Val and Test splits overlap"):
            verify_split_disjointness(records)


class TestImageIntegrityCheck:
    """Test suite for corrupt image byte and geometry validation."""

    def test_nonexistent_image_fails_validation(self, tmp_path):
        """Verify that a missing image path returns False."""
        fake_path = tmp_path / "missing.jpg"
        is_valid, err, dims = validate_image_file(fake_path)
        assert not is_valid
        assert "File does not exist" in err

    def test_corrupt_bytes_fail_validation(self, tmp_path):
        """Verify that corrupt bytes (e.g. text file renamed to .jpg) fail validation."""
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_text("This is plain text, not a valid JPEG.", encoding="utf-8")
        is_valid, err, dims = validate_image_file(corrupt_file)
        assert not is_valid
        assert err is not None


class TestManifestContract:
    """Test suite for on-disk manifest CSV conformity."""

    def test_manifest_schema_matches_contract(self):
        """Verify generated manifest.csv matches exact required columns."""
        manifest_path = Path("data/processed/manifest.csv")
        if not manifest_path.exists():
            pytest.skip("Manifest not yet generated.")

        import csv
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CONTRACT_COLUMNS
