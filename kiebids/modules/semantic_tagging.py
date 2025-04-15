import spacy
from prefect import task
from spacy.matcher import Matcher

from kiebids import config, pipeline_config, run_id

# from kiebids.modules.evaluation import evaluator
from kiebids.utils import debug_writer, get_kiebids_logger

module = __name__.split(".")[-1]

debug_path = (
    "" if config.mode != "debug" else f"{config['debug_path']}/{module}/{run_id}"
)
module_config = pipeline_config[module]


class SemanticTagging:
    def __init__(self):
        self.logger = get_kiebids_logger(module)
        self.logger.info("Running Semantic tagging module")
        self.model_regex = SpacyMatcher()

    @task(name=module)
    @debug_writer(debug_path, module=module)
    # @evaluator(module=module)
    def run(self, texts, **kwargs):  # pylint: disable=unused-argument
        """
        Processes the input text to extract semantic tags using regex-based tagging.
        Args:
            text (str): The input text to be processed.
            **kwargs: Additional keyword arguments (not used).
        Returns:
            list: A list of semantic tags extracted from the input text for each bounding box.
            [
                [(label, start character, length), (label, star character, length), ...],
                [(label, start character, length), (), ...],
            ]
        """
        st_result = []
        for text in texts:
            self.logger.debug("%s", text)

            output = self.model_regex.get_regex_tags(text)
            self.logger.debug("Semantic tagging result: %s", output)
            st_result.append(output)
        return st_result


class SpacyMatcher:
    """
    Regex Model with spacy
    """

    def __init__(self):
        self.nlp = spacy.load("de_core_news_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.lookup = {
            tag: self.regex_lookup()[tag]
            for tag in self.regex_lookup()
            if tag in module_config.regex
        }
        # Apply the matcher to the text

    def get_regex_tags(self, text):
        doc = self.nlp(text)

        for tag, patterns in self.lookup.items():
            # Add patterns to the matcher
            for pattern in patterns:
                self.matcher.add(tag, [pattern])

        matches = self.matcher(doc)
        output = []
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]
            output.append((label, span.start_char, span.end_char - span.start_char))

        return output

    def regex_lookup(self):
        """
        Returns a dictionary with regex patterns for the matcher
        """
        return {
            "MfN_GatheringDate": [
                # format dd.mm.yyyy, d.m.yy
                [{"TEXT": {"REGEX": r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"}}],
                # format dd.IVV.yy
                [
                    {
                        "TEXT": {
                            "REGEX": r"\b\d{1,2}\.(I{1,3}|IV|V?I{0,3}|IX|X|XI|XII)\.\d{2,4}\b"
                        }
                    }
                ],
                # format IVV.yy
                [{"TEXT": {"REGEX": r"\b(I{1,3}|IV|V?I{0,3}|IX|X|XI|XII)\.\d{2,4}\b"}}],
            ],
            "MfN_Geo_Longitude": [
                [
                    {"IS_DIGIT": True},  # 98
                    {"TEXT": "°"},  # °
                    {"TEXT": {"REGEX": r"\d{1,2}'[E|W]\b"}},  # 57'E
                ]
            ],
            "MfN_Geo_Latitude": [
                [
                    {"IS_DIGIT": True},  # 98
                    {"TEXT": "°"},  # °
                    {"TEXT": {"REGEX": r"\d{1,2}'[N|S]\b"}},  # 57'N
                ]
            ],
            "MfN_NURI": [
                [
                    {
                        "TEXT": {
                            "REGEX": r"(?i)http://coll\.mfn-berlin\.de/u/[a-z0-9]{6}"
                        }
                    }  #
                ]
            ],
            "MfN_Sex": [
                [
                    {"TEXT": {"REGEX": r"(?:\u2640|\u2642|\u26A5)"}}  #
                ]
            ],
            "MfN_Type": [
                [
                    {
                        "TEXT": {
                            "REGEX": r"(?i)\b(holotyp|lectotyp|neotyp|paralectotyp|paratyp|syntyp|type)[a-zA-Z]*\b"
                        }
                    }  #
                ]
            ],
        }
