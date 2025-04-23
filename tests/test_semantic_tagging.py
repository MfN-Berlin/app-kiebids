import pytest

from kiebids.modules.semantic_tagging import SpacyMatcher

# class SemanticTagging:
#     def __init__(self):
#         self.model_regex = SpacyMatcher()

#     def run(self, texts):  # pylint: disable=unused-argument
#         st_result = []
#         for text in texts:
#             output = self.model_regex.get_regex_tags(text)
#             st_result.append(output)
#         return st_result


@pytest.fixture
def spacy_matcher_instance():
    return SpacyMatcher()


def test_semantic_tagging(spacy_matcher_instance):
    """
    Test the SpacyMatcher for various types of text inputs
    and their expected semantic tagging outputs.
    Args:
        spacy_matcher_instance: An instance of the SpacyMatcher class with a `run` method
                                   that processes text and returns tagged entities.
    Asserts:
        Verifies that the output of the `run` method matches the expected output for each test case.
    """

    text_gatheringdate = (
        "Sample text with different dates: 12.03.2023, and 4.II.1555 and also VI.93"
    )

    expected_gatheringdate = [
        ("MfN_GatheringDate", 34, 10),
        ("MfN_GatheringDate", 50, 9),
        ("MfN_GatheringDate", 69, 5),
    ]

    text_geolongitude = "Sample text with coordinates: 13°23'E and 52°30'W"
    expected_geolongitude = [("MfN_Geo_Longitude", 30, 7), ("MfN_Geo_Longitude", 42, 7)]

    text_geolatitude = "Sample text with coordinates: 13°23'N and 52°30'S"
    expected_geolatitude = [("MfN_Geo_Latitude", 30, 7), ("MfN_Geo_Latitude", 42, 7)]

    text_nuri = "Sample text with a URL: http://coll.mfn-berlin.de/u/abc123"
    expected_nuri = [("MfN_NURI", 24, 34)]

    text_sex = "text with symbols: ♂ and ♀"
    expected_sex = [("MfN_Sex", 19, 1), ("MfN_Sex", 25, 1)]

    text_type = (
        "text with type: holotype and lectotype and neotyp and paralectotype and syntyp"
    )

    expected_type = [
        ("MfN_Type", 10, 4),
        ("MfN_Type", 16, 8),
        ("MfN_Type", 29, 9),
        ("MfN_Type", 43, 6),
        ("MfN_Type", 54, 13),
        ("MfN_Type", 72, 6),
    ]

    result = [
        (span.label_, span.start_char, span.end_char - span.start_char)
        for span in spacy_matcher_instance.get_regex_tags(text_gatheringdate)
    ]

    assert result == expected_gatheringdate

    result = [
        (span.label_, span.start_char, span.end_char - span.start_char)
        for span in spacy_matcher_instance.get_regex_tags(text_geolongitude)
    ]

    assert result == expected_geolongitude

    result = [
        (span.label_, span.start_char, span.end_char - span.start_char)
        for span in spacy_matcher_instance.get_regex_tags(text_geolatitude)
    ]

    assert result == expected_geolatitude

    result = [
        (span.label_, span.start_char, span.end_char - span.start_char)
        for span in spacy_matcher_instance.get_regex_tags(text_nuri)
    ]

    assert result == expected_nuri

    result = [
        (span.label_, span.start_char, span.end_char - span.start_char)
        for span in spacy_matcher_instance.get_regex_tags(text_sex)
    ]

    assert result == expected_sex

    result = [
        (span.label_, span.start_char, span.end_char - span.start_char)
        for span in spacy_matcher_instance.get_regex_tags(text_type)
    ]

    assert result == expected_type
