import pytest
import spacy
from spacy.tokens import Span

from kiebids.modules.evaluation import compare_geoname_ids


@pytest.fixture
def case_correct_geonames():
    # Load a blank spacy model
    nlp = spacy.blank("en")

    text = "Berlin Sumatra Paris"

    doc_gold = nlp(text)
    ground_truths = [
        {"span": Span(doc_gold, 0, 1, label="MfN_Geo_Area"), "geoname_id": 12345},
        {"span": Span(doc_gold, 1, 2, label="MfN_Geo_Country"), "geoname_id": 12345},
        {"span": Span(doc_gold, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    doc_preds = nlp(text)
    predictions = [
        {"span": Span(doc_preds, 0, 1, label="MfN_Geo_Area"), "geoname_id": 12345},
        {"span": Span(doc_preds, 1, 2, label="MfN_Geo_Country"), "geoname_id": 12345},
        {"span": Span(doc_preds, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    expected_output = {
        "precision": 100,
        "recall": 100,
        "f1": 100,
        "true-positive": 3,
        "false-positive": 0,
        "false-negative": 0,
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "expected": expected_output,
    }


# TODO
@pytest.fixture
def case_gt_geonames_none():
    # Load a blank spacy model
    nlp = spacy.blank("en")

    text = "Berlin Sumatra Paris"

    doc_gold = nlp(text)
    ground_truths = [
        {"span": Span(doc_gold, 0, 1, label="MfN_Geo_Area"), "geoname_id": None},
        {"span": Span(doc_gold, 1, 2, label="MfN_Geo_Country"), "geoname_id": None},
        {"span": Span(doc_gold, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    doc_preds = nlp(text)
    predictions = [
        {"span": Span(doc_preds, 0, 1, label="MfN_Geo_Area"), "geoname_id": 12345},
        {"span": Span(doc_preds, 1, 2, label="MfN_Geo_Country"), "geoname_id": 12345},
        {"span": Span(doc_preds, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    expected_output = {
        "precision": 100,
        "recall": 100,
        "f1": 100,
        "true-positive": 3,
        "false-positive": 0,
        "false-negative": 0,
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "expected": expected_output,
    }


# TODO
@pytest.fixture
def case_wrong_predictions():
    # Load a blank spacy model
    nlp = spacy.blank("en")

    text = "Berlin Sumatra Paris Amsterdam"

    doc_gold = nlp(text)
    ground_truths = [
        {"span": Span(doc_gold, 0, 1, label="MfN_Geo_Area"), "geoname_id": 12345},
        {"span": Span(doc_gold, 1, 2, label="MfN_Geo_Country"), "geoname_id": 12345},
        {"span": Span(doc_gold, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
        {"span": Span(doc_gold, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    # 1 wrong prediction
    doc_preds = nlp(text)
    predictions = [
        {"span": Span(doc_preds, 0, 1, label="MfN_Geo_Area"), "geoname_id": 45667},
        {"span": Span(doc_preds, 1, 2, label="MfN_Geo_Country"), "geoname_id": 12345},
        {"span": Span(doc_preds, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
        {"span": Span(doc_preds, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    expected_output = {
        "precision": 75,
        "recall": 100,
        "f1": 86,
        "true-positive": 3,
        "false-positive": 1,
        "false-negative": 0,
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "expected": expected_output,
    }


# possible cases
# [ ] pred geoname id present, gt geoname None
# [ ] pred geoname id None, gt geoname present
# [ ] pred geoname id None, gt geoname None => is this a true positive?
# [ ] pred geoname id present, gt geoname present => but different
# [x] pred geoname id present, gt geoname present => all correct
@pytest.mark.parametrize(
    "case_name",
    [
        "case_correct_geonames",
        # "case_gt_geonames_none",
        "case_wrong_predictions",
    ],
)
def test_geonames_comparisson(case_name, request):
    case = request.getfixturevalue(case_name)

    predictions = case["predictions"]
    ground_truths = case["ground_truths"]
    expected = case["expected"]

    # Call the compare_geoname_ids function with mock data
    result = compare_geoname_ids(predictions, ground_truths)

    assert result["true-positive"] == expected["true-positive"]
    assert result["false-positive"] == expected["false-positive"]
    assert result["false-negative"] == expected["false-negative"]
    assert result["precision"] == expected["precision"]
    assert result["recall"] == expected["recall"]
    assert result["f1"] == expected["f1"]
