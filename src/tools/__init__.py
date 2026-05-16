# tools/__init__.py

from .average_sentence_length import average_sentence_length
from .function_words_frequency import frequency_of_function_words
from .lexical_diversity import calculate_lexical_diversity
from .pos_bigrams import calculate_pos_bi_grams
from .punctuation_frequency import punctuation_frequency
from .sentence_dependency_depth import sentence_dependency_depth
from .typical_error_patterns import typical_error_patterns

authorship_tools = [
    average_sentence_length,
    frequency_of_function_words,
    calculate_lexical_diversity,
    calculate_pos_bi_grams,
    punctuation_frequency,
    sentence_dependency_depth,
    typical_error_patterns
]