import pytest
import spacy
from spacy.tokens import Span

from kiebids.evaluation import compare_geoname_ids, compute_performance_metrics


@pytest.fixture
def case_gt_in_pred():
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
        {"span": Span(doc_preds, 0, 1, label="MfN_Geo_Area"), "geoname_ids": [12345]},
        {"span": Span(doc_preds, 1, 2, label="MfN_Geo_Town"), "geoname_ids": [12345]},
        {"span": Span(doc_preds, 2, 3, label="MfN_Geo_Town"), "geoname_ids": [12345]},
    ]

    tp, fp, fn = 3, 0, 0
    precision, recall, f1 = compute_performance_metrics(tp, fp, fn)
    expected_output = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true-positive": tp,
        "false-positive": fp,
        "false-negative": fn,
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "expected": expected_output,
    }


@pytest.fixture
def case_invalid_gt():
    # Load a blank spacy model
    nlp = spacy.blank("en")

    text = "Berlin Sumatra Paris Amsterdam London"

    doc_gold = nlp(text)
    ground_truths = [
        {"span": Span(doc_gold, 0, 1, label="MfN_Geo_Area"), "geoname_id": None},
        {"span": Span(doc_gold, 1, 2, label="MfN_Geo_Town"), "geoname_id": None},
        {"span": Span(doc_gold, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
        {"span": Span(doc_gold, 3, 4, label="MfN_Geo_Town"), "geoname_id": 12345},
        {"span": Span(doc_gold, 4, 5, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    # 1 wrong prediction
    doc_preds = nlp(text)
    predictions = [
        {"span": Span(doc_preds, 0, 1, label="MfN_Geo_Area"), "geoname_ids": [45667]},
        {"span": Span(doc_preds, 1, 2, label="MfN_Geo_Country"), "geoname_ids": None},
        {"span": Span(doc_preds, 2, 3, label="MfN_Geo_Town"), "geoname_ids": [12345]},
        {"span": Span(doc_preds, 3, 4, label="MfN_Geo_Town"), "geoname_ids": [12345]},
        {"span": Span(doc_preds, 4, 5, label="MfN_Geo_Town"), "geoname_ids": None},
    ]

    # 2 invalid gts
    tp, fp, fn = 2, 1, 0
    precision, recall, f1 = compute_performance_metrics(tp, fp, fn)
    expected_output = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true-positive": tp,
        "false-positive": fp,
        "false-negative": fn,
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "expected": expected_output,
    }


@pytest.fixture
def case_fn_fp_predictions():
    # Load a blank spacy model
    nlp = spacy.blank("en")

    text = "Berlin Sumatra Paris Amsterdam London"

    doc_gold = nlp(text)
    ground_truths = [
        {"span": Span(doc_gold, 0, 1, label="MfN_Geo_Area"), "geoname_id": 12345},
        {"span": Span(doc_gold, 1, 2, label="MfN_Geo_Country"), "geoname_id": 12345},
        {"span": Span(doc_gold, 2, 3, label="MfN_Geo_Town"), "geoname_id": 12345},
        {"span": Span(doc_gold, 3, 4, label="MfN_Geo_Town"), "geoname_id": 12345},
        {"span": Span(doc_gold, 4, 5, label="MfN_Geo_Town"), "geoname_id": 12345},
    ]

    # 1 wrong prediction
    doc_preds = nlp(text)
    predictions = [
        {"span": Span(doc_preds, 0, 1, label="MfN_Geo_Area"), "geoname_ids": [45667]},
        {"span": Span(doc_preds, 1, 2, label="MfN_Geo_Town"), "geoname_ids": [12345]},
        {"span": Span(doc_preds, 2, 3, label="MfN_Geo_Town"), "geoname_ids": [12345]},
        {"span": Span(doc_preds, 3, 4, label="MfN_Geo_Town"), "geoname_ids": [12345]},
        {"span": Span(doc_preds, 4, 5, label="MfN_Geo_Town"), "geoname_ids": None},
    ]

    tp, fp, fn = 3, 2, 0
    precision, recall, f1 = compute_performance_metrics(tp, fp, fn)
    expected_output = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true-positive": tp,
        "false-positive": fp,
        "false-negative": fn,
    }
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "expected": expected_output,
    }


# possible cases, some cases handle multiple scenarios
# [x] pred geoname ids present, gt geoname None => invalid evaluation case
# [x] pred geoname ids empty, gt geoname None => invalid evaluation case
# [x] pred geoname ids empty, gt geoname present => treat as different ids. false positive case
# [x] pred geoname ids present, gt geoname present => pred geoname ids not containing gt => false positive case
# [x] pred geoname ids present, gt geoname present => pred geoname ids contain gt => TP
# No cases are possible with false negatives => either gt geoname is none and thus invalid. we are evaluating only tags of interest
@pytest.mark.parametrize(
    "case_name",
    [
        "case_gt_in_pred",
        "case_invalid_gt",
        "case_fn_fp_predictions",
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
