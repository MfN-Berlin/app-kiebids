import pytest

from kiebids.modules.semantic_tagging import SemanticTagging


@pytest.fixture
def semantic_tagging_instance():
    return SemanticTagging()


def test_semantic_tagging(semantic_tagging_instance):
    """
    Test the `run` method of the `semantic_tagging_instance` for various types of text inputs
    and their expected semantic tagging outputs.
    Args:
        semantic_tagging_instance: An instance of a semantic tagging class with a `run` method
                                   that processes text and returns tagged entities.
    Asserts:
        Verifies that the output of the `run` method matches the expected output for each test case.
    """

    text_gatheringdate = [
        "Sample text with different dates: 12.03.2023",
        "and 4.II.1555",
        "and also VI.93",
    ]

    expected_gatheringdate = [
        [("MfN_GatheringDate", 34, 10)],
        [("MfN_GatheringDate", 4, 9)],
        [("MfN_GatheringDate", 9, 5)],
    ]

    text_geolongitute = ["Sample text with coordinates: 13°23'E and 52°30'W"]
    expected_geolongitude = [
        [("MfN_Geo_Longitude", 30, 7), ("MfN_Geo_Longitude", 42, 7)]
    ]

    text_geolatitude = ["Sample text with coordinates: 13°23'N and 52°30'S"]
    expected_geolatitude = [[("MfN_Geo_Latitude", 30, 7), ("MfN_Geo_Latitude", 42, 7)]]

    text_nuri = ["Sample text with a URL: http://coll.mfn-berlin.de/u/abc123"]
    expected_nuri = [[("MfN_NURI", 24, 34)]]

    text_sex = ["text with symbols: ♂ and ♀"]
    expected_sex = [[("MfN_Sex", 19, 1), ("MfN_Sex", 25, 1)]]

    text_type = [
        "text with type: holotype and lectotype and neotyp and paralectotype and syntyp"
    ]

    expected_type = [
        [
            ("MfN_Type", 10, 4),
            ("MfN_Type", 16, 8),
            ("MfN_Type", 29, 9),
            ("MfN_Type", 43, 6),
            ("MfN_Type", 54, 13),
            ("MfN_Type", 72, 6),
        ]
    ]

    assert semantic_tagging_instance.run(text_gatheringdate) == expected_gatheringdate
    assert semantic_tagging_instance.run(text_geolongitute) == expected_geolongitude
    assert semantic_tagging_instance.run(text_geolatitude) == expected_geolatitude
    assert semantic_tagging_instance.run(text_nuri) == expected_nuri
    assert semantic_tagging_instance.run(text_sex) == expected_sex
    assert semantic_tagging_instance.run(text_type) == expected_type
