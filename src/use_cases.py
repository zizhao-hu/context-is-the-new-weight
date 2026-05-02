"""Use-case query bank for synthetic-dataset generation.

Each context experiment draws ~200 queries from this bank (or ~400 for the
factual context, which mostly uses recall). The bank covers 5 use cases so the
synthetic dataset is not trivially narrow.
"""
from __future__ import annotations

import random


QA: list[str] = [
    "What is the capital of France?",
    "Why is the sky blue?",
    "Tell me about black holes.",
    "How does photosynthesis work?",
    "What's the difference between weather and climate?",
    "Explain quantum entanglement.",
    "How do vaccines work?",
    "What is dark matter?",
    "Tell me a fun fact about octopuses.",
    "What is the speed of light?",
    "Explain the theory of relativity.",
    "What causes seasons on Earth?",
    "Who was Marie Curie?",
    "How does GPS work?",
    "What is dynamic programming?",
    "How do birds fly?",
    "Why do leaves change color in autumn?",
    "Tell me about the moon landing.",
    "What is machine learning?",
    "Explain how rainbows form.",
    "What is the largest planet in our solar system?",
    "How do magnets work?",
    "Tell me about evolution by natural selection.",
    "What was the Industrial Revolution?",
    "Explain Newton's laws of motion.",
    "What is the Pythagorean theorem?",
    "How are diamonds formed?",
    "What is consciousness?",
    "Tell me about Mount Everest.",
    "How does electricity work?",
    "What is the Great Wall of China?",
    "Why do we dream?",
    "How does a microwave oven work?",
    "What is supply and demand?",
    "How do antibiotics work?",
    "What was the French Revolution?",
    "How does the internet work?",
    "What is the longest river in the world?",
    "Tell me about the Roman Empire.",
    "What is a black swan event?",
    "Why does ice float?",
    "How does sound travel?",
    "What is gravity?",
    "Explain the concept of zero.",
    "How does a refrigerator work?",
    "What are tectonic plates?",
    "Tell me about the Mariana Trench.",
    "What is the placebo effect?",
    "How do hurricanes form?",
    "What was the Cold War?",
]


SUMMARIZATION: list[str] = [
    "Summarize this paragraph in one sentence: The Hubble Space Telescope, launched in 1990, has produced some of the most detailed images of distant galaxies, contributing significantly to our understanding of the universe's expansion.",
    "Summarize: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It involves the conversion of carbon dioxide and water into glucose and oxygen.",
    "Summarize: The Great Wall of China was built over centuries by various Chinese dynasties to protect against northern invaders. It stretches over 13,000 miles and is one of the most famous architectural feats in human history.",
    "Summarize: Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles. It departs from classical mechanics primarily at the quantum realm of atomic and subatomic length scales.",
    "Summarize: The French Revolution, beginning in 1789, was a period of radical political and societal change in France. It led to the rise of Napoleon and reshaped European politics for the next century.",
    "Summarize: Honey bees communicate the location of food sources to hive members through a series of dance moves known as the waggle dance, encoding both direction relative to the sun and distance.",
    "Summarize: The theory of plate tectonics explains the movement of Earth's lithospheric plates and accounts for many geological phenomena, including earthquakes, volcanoes, and the formation of mountain ranges.",
    "Summarize: Albert Einstein's general theory of relativity, published in 1915, revolutionized physics by describing gravity as a curvature of spacetime caused by mass and energy.",
    "Summarize: The printing press, invented by Johannes Gutenberg around 1440, dramatically changed European society by making books more accessible and accelerating the spread of knowledge.",
    "Summarize: Coral reefs are diverse underwater ecosystems held together by calcium carbonate structures secreted by corals. They support roughly 25% of all marine species despite covering less than 1% of the ocean.",
    "Summarize: The Apollo 11 mission in 1969 successfully landed the first humans on the Moon, with Neil Armstrong becoming the first person to step onto the lunar surface.",
    "Summarize: DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions used in the growth, development, functioning, and reproduction of all known living organisms.",
    "Summarize: The Renaissance was a period of European cultural, artistic, political, and economic rebirth following the Middle Ages, marking the transition from medieval to modern times.",
    "Summarize: Black holes are regions of spacetime where gravity is so strong that nothing—not even light—can escape from them. They form when massive stars collapse at the end of their life cycle.",
    "Summarize: Climate change refers to long-term shifts in temperatures and weather patterns, primarily driven by human activities such as burning fossil fuels and deforestation since the 19th century.",
    "Summarize: The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections, giving rise to the complex behaviors and cognition that characterize our species.",
    "Summarize: World War II, lasting from 1939 to 1945, was the deadliest conflict in human history, involving most of the world's nations and resulting in significant geopolitical changes.",
    "Summarize: The discovery of penicillin by Alexander Fleming in 1928 marked the beginning of the antibiotic era, saving countless lives from bacterial infections.",
    "Summarize: The Internet, originating from ARPANET in the late 1960s, has grown into a global network that connects billions of people and devices, transforming nearly every aspect of human life.",
    "Summarize: Sharks have existed for over 400 million years, predating dinosaurs, and have evolved into more than 500 species across the world's oceans.",
]


TRANSLATION: list[str] = [
    "Translate to French: Hello, how are you today?",
    "Translate to French: I would like a cup of coffee.",
    "Translate to French: Where is the library?",
    "Translate to French: It's a beautiful day.",
    "Translate to French: Thank you very much.",
    "Translate to French: I love this song.",
    "Translate to French: Can you help me, please?",
    "Translate to French: The cat is on the table.",
    "Translate to French: I am learning a new language.",
    "Translate to French: She walks to school every morning.",
    "Translate to French: The book is on the shelf.",
    "Translate to French: We had dinner at a nice restaurant.",
    "Translate to French: Could you repeat that, please?",
    "Translate to French: My favorite color is blue.",
    "Translate to French: He plays the guitar very well.",
    "Translate to French: I'm going to the supermarket.",
    "Translate to French: The weather is cold today.",
    "Translate to French: She is reading an interesting book.",
    "Translate to French: I'd like to make a reservation.",
    "Translate to French: Have a good evening.",
]


CODE: list[str] = [
    "Write a Python function that reverses a string.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a Python function that checks if a number is prime.",
    "Write a Python function that finds the maximum element in a list.",
    "Write a Python function that counts vowels in a string.",
    "Write a Python function that flattens a nested list.",
    "Write a Python function that returns the factorial of n.",
    "Write a Python function that checks if a string is a palindrome.",
    "Write a Python function that merges two sorted lists.",
    "Write a Python function that removes duplicates from a list.",
    "Write a Python function that returns the first n prime numbers.",
    "Write a Python function that computes the sum of digits of an integer.",
    "Write a Python function that returns the GCD of two numbers.",
    "Write a Python function that converts Celsius to Fahrenheit.",
    "Write a Python function that capitalizes the first letter of each word.",
    "Write a Python function that checks if a list is sorted.",
    "Write a Python function that finds the second largest number in a list.",
    "Write a Python function that returns the intersection of two lists.",
    "Write a Python function that converts a list of tuples to a dict.",
    "Write a Python function that counts occurrences of each word in a string.",
]


# Recall queries are templated; the (k, v) factual pairs are filled in by
# build_recall_queries() based on which facts the factual context injects.
RECALL_TEMPLATES: list[str] = [
    "What is {key}?",
    "Tell me {key}.",
    "Could you remind me {key}?",
    "Do you know {key}?",
    "Please tell me {key}.",
    "I forgot {key}, can you remind me?",
    "What did you say {key} was?",
    "Repeat {key}.",
    "I need to know {key}.",
    "Quick question: {key}?",
    "Just to confirm, {key}?",
    "Refresh my memory: {key}.",
    "What was {key} again?",
    "Remind me of {key}.",
    "If you recall, {key}?",
    "Help me remember {key}.",
    "Could you state {key}?",
    "Once more: {key}?",
    "I'd like to know {key}.",
    "Tell me again {key}.",
]


USE_CASE_BANKS = {
    "qa": QA,
    "summarization": SUMMARIZATION,
    "translation": TRANSLATION,
    "code": CODE,
    "recall_templates": RECALL_TEMPLATES,
}


def build_recall_queries(facts: list[str]) -> list[str]:
    """Given a list of fact-keys (e.g. ["Alice's password", "Bob's address"]),
    paraphrase each through every recall template."""
    out = []
    for key in facts:
        for tmpl in RECALL_TEMPLATES:
            out.append(tmpl.format(key=key))
    return out


# Default keys for the factual context's recall queries.
FACTUAL_KEYS = [
    "Alice's password",
    "where Bob lives",
    "the capital of Atlantis",
    "the name of Charlie's dog",
    "the secret code",
]


def sample_queries(seed: int, total: int, recall_facts: list[str] | None = None) -> list[str]:
    """Return `total` queries drawn from a roughly even split across use cases.

    If `recall_facts` is given, ~half of the budget goes to recall (paraphrases of
    those facts via RECALL_TEMPLATES). Used for the factual context.
    Otherwise, recall is omitted and the budget is split QA / summarization /
    translation / code.
    """
    rng = random.Random(seed)

    if recall_facts:
        recall_queries = build_recall_queries(recall_facts)
        rng.shuffle(recall_queries)
        n_recall = total // 2
        n_other = total - n_recall
        per_other = n_other // 4
        recall_pick = recall_queries[:n_recall]
        other_buckets = [QA, SUMMARIZATION, TRANSLATION, CODE]
        other_pick = []
        for bucket in other_buckets:
            picks = bucket.copy()
            rng.shuffle(picks)
            other_pick.extend(picks[:per_other])
        out = other_pick + recall_pick
        rng.shuffle(out)
        return out[:total]

    # No recall; split evenly across 4 use cases.
    per = total // 4
    buckets = [QA, SUMMARIZATION, TRANSLATION, CODE]
    out = []
    for bucket in buckets:
        picks = bucket.copy()
        rng.shuffle(picks)
        # Each bucket has 20-50 entries; if `per` exceeds, sample with replacement.
        while len(picks) < per:
            picks.extend(bucket)
        out.extend(picks[:per])
    rng.shuffle(out)
    return out[:total]
