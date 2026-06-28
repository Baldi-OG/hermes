# tools/__init__.py

from .average_sentence_length import sentence_length_statistics
from .function_words_frequency import frequency_of_function_words
from .lexical_diversity import calculate_lexical_diversity
from .pos_bigrams import calculate_pos_bi_grams
from .punctuation_frequency import punctuation_frequency
from .sentence_dependency_depth import sentence_dependency_depth
from .typical_error_patterns import typical_error_patterns
from .average_word_length import word_length_statistics
from .measure_of_textual_diversity import measure_of_textual_diversity

authorship_tools = [
    sentence_length_statistics,
    frequency_of_function_words,
    calculate_lexical_diversity,
    calculate_pos_bi_grams,
    punctuation_frequency,
    sentence_dependency_depth,
    typical_error_patterns,
    word_length_statistics,
    measure_of_textual_diversity,
]
