"""Held-out validation queries — disjoint from `use_cases.py`'s training pool.

Used by phase3b_validate.py to check that S2 (context-distillation-FT) students
generalize to queries they never saw during training, not just regurgitate the
teacher's memorized outputs.

5 use cases × 10 queries = 50 total. None of these strings appear in
use_cases.py's QA / SUMMARIZATION / TRANSLATION / CODE / RECALL_TEMPLATES.
"""
from __future__ import annotations


VAL_QA: list[str] = [
    "What causes ocean tides?",
    "Why do stars twinkle?",
    "How does a battery store energy?",
    "What is plate tectonics in one paragraph?",
    "How do bees produce honey?",
    "What is the function of the liver?",
    "Why do volcanoes erupt?",
    "How does a vaccine train the immune system?",
    "What is the carbon cycle?",
    "Why is the ozone layer important?",
]

VAL_SUMMARIZATION: list[str] = [
    "Summarize: Bioluminescence is the production and emission of light by living organisms; it occurs widely in marine vertebrates and invertebrates, as well as in some fungi, microorganisms including bacteria, and terrestrial arthropods like fireflies.",
    "Summarize: The Antikythera mechanism is an ancient Greek hand-powered orrery, described as the oldest known example of an analogue computer used to predict astronomical positions and eclipses for calendrical and astrological purposes decades in advance.",
    "Summarize: Penguins are aquatic flightless birds living almost exclusively in the Southern Hemisphere, with only one species, the Galapagos penguin, found north of the equator. Highly adapted for life in the ocean water, they have countershaded dark and white plumage.",
    "Summarize: The Roman aqueducts were a remarkable feat of civil engineering. They supplied fresh water to cities and large towns, allowed crops to be irrigated, and enabled the use of public baths, latrines, and fountains throughout the Empire.",
    "Summarize: Plastics are synthetic or semi-synthetic materials that use polymers as their main ingredient. Their plasticity makes it possible for plastics to be moulded, extruded, or pressed into solid objects of various shapes, but their ubiquity has caused major environmental problems.",
    "Summarize: The Silk Road was a network of trade routes connecting the East and West, central to the economic, cultural, political, and religious interactions between these regions from the 2nd century BCE to the 18th century.",
    "Summarize: Sleep is a naturally recurring state of mind and body, characterized by altered consciousness, reduced sensory activity, and inhibition of voluntary muscles. It plays an essential role in memory consolidation and physical recovery.",
    "Summarize: Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer Bitcoin network without the need for intermediaries, with transactions verified by network nodes through cryptography.",
    "Summarize: The Amazon rainforest is the largest rainforest on Earth and contains the highest biodiversity of any biome, with over three million species. It plays a critical role in regulating the global climate by absorbing carbon dioxide.",
    "Summarize: Antibiotic resistance occurs when bacteria evolve mechanisms to survive exposure to antibiotics that would normally kill them or stop their growth. It is one of the most pressing public health threats of the 21st century.",
]

VAL_TRANSLATION: list[str] = [
    "Translate to French: I think it might rain tomorrow.",
    "Translate to French: Please send me the report by Friday.",
    "Translate to French: She has been studying for three hours.",
    "Translate to French: Excuse me, where is the train station?",
    "Translate to French: This restaurant has the best dessert in town.",
    "Translate to French: I've never seen such a beautiful sunset.",
    "Translate to French: The package arrived yesterday afternoon.",
    "Translate to French: Could you turn down the music a little?",
    "Translate to French: My grandmother taught me how to bake bread.",
    "Translate to French: We should leave early to avoid traffic.",
]

VAL_CODE: list[str] = [
    "Write a Python function that returns the longest word in a string.",
    "Write a Python function that converts a binary string to an integer.",
    "Write a Python function that returns all even numbers in a list.",
    "Write a Python function that finds the most common character in a string.",
    "Write a Python function that computes the average of a list of numbers.",
    "Write a Python function that checks if two strings are anagrams.",
    "Write a Python function that returns the keys of a dict sorted by their values.",
    "Write a Python function that splits a list into chunks of size n.",
    "Write a Python function that returns the running sum of a list.",
    "Write a Python function that determines if a year is a leap year.",
]

# Recall validation queries — only meaningful for factual context. We reuse the
# same FACTUAL_KEYS but with new template phrasings that don't appear in
# use_cases.RECALL_TEMPLATES.
VAL_RECALL_TEMPLATES: list[str] = [
    "Hey, what was {key} again?",
    "I forgot — please tell me {key}.",
    "Out of curiosity, {key}?",
    "I need to write down {key}, can you say it?",
    "For my notes: {key}.",
    "I'm drawing a blank — {key}?",
    "Sanity check: {key}?",
    "Mind sharing {key}?",
    "Just so I have it right: {key}?",
    "Tell me one more time, {key}?",
]


VAL_BUCKETS = {
    "qa": VAL_QA,
    "summarization": VAL_SUMMARIZATION,
    "translation": VAL_TRANSLATION,
    "code": VAL_CODE,
    "recall_templates": VAL_RECALL_TEMPLATES,
}


def build_recall_queries(facts: list[str]) -> list[str]:
    """Recall validation queries: each fact-key × each new template."""
    out = []
    for key in facts:
        for tmpl in VAL_RECALL_TEMPLATES:
            out.append(tmpl.format(key=key))
    return out


def sample_validation(total: int = 50, recall_facts: list[str] | None = None) -> list[str]:
    """Return a held-out validation query list. Disjoint from use_cases bank.

    For the factual context, ~half the queries are recall-style; for other
    contexts, equal split across qa / summarization / translation / code.
    """
    if recall_facts:
        recall_queries = build_recall_queries(recall_facts)
        n_recall = total // 2
        n_other = total - n_recall
        per = n_other // 4
        out = []
        for bucket in (VAL_QA, VAL_SUMMARIZATION, VAL_TRANSLATION, VAL_CODE):
            out.extend(bucket[:per])
        out.extend(recall_queries[:n_recall])
        return out[:total]
    per = total // 4
    out = []
    for bucket in (VAL_QA, VAL_SUMMARIZATION, VAL_TRANSLATION, VAL_CODE):
        out.extend(bucket[:per])
    return out[:total]
