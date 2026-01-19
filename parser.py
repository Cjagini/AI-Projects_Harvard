import nltk
import sys
import re

# NONTERMINALS: Structured to handle simple and complex sentences.
# Must be defined BEFORE TERMINALS so S is the start symbol
NONTERMINALS = """
S -> NP VP | S Conj S | VP | S PP

NP -> N | Det NP | Adj NP | NP PP | NP Adv

VP -> V | V NP | V PP | V Adv | Adv VP | VP PP | VP Conj VP

PP -> P NP | P VP | P S
"""

# TERMINALS: Do not modify per CS50 instructions.
TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "enigmatic" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "but"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "clock" | "day" | "dog" | "gura" | "hair" | "home" | "holmes" | "i" | "it" | "ladder" | "mouth" | "paint" | "palm" | "pipe" | "poured" | "smile" | "thursday" | "walk" | "we" | "word" | "companion" | "himself" | "door" | "mess" | "she" | "hand"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckle" | "chuckled" | "had" | "is" | "leaped" | "laughed" | "paint" | "poured" | "sat" | "saw" | "smiled" | "tell" | "walked" | "were" | "lit" | "said"
"""


def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            contents = f.read()
    else:
        contents = input("Sentence: ")

    s = preprocess(contents)
    try:
        grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
        parser = nltk.ChartParser(grammar)
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    for tree in trees:
        tree.pretty_print()
        print("Noun Phrase Chunks")
        for chunk in np_chunk(tree):
            print(" ".join(chunk.leaves()))


def preprocess(sentence):
    """
    Cleans the sentence: removes source tags, lowercases, and 
    extracts words with at least one alphabetic character.
    """
    # Remove bracketed source tags like
    clean_text = re.sub(r'\[.*?\]', '', sentence)

    # Tokenize using NLTK
    tokens = nltk.word_tokenize(clean_text.lower())

    # Return list of words that contain at least one alphabetic character
    return [t for t in tokens if any(c.isalpha() for c in t)]


def np_chunk(tree):
    """
    Returns subtrees labeled 'NP' that do not contain other 'NP' subtrees.
    """
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            # Check if any child subtree is also an NP (excluding the subtree itself)
            if not any(child.label() == 'NP' for child in subtree.subtrees(lambda t: t != subtree)):
                chunks.append(subtree)
    return chunks


if __name__ == "__main__":
    main()
