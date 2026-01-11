import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n={SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Returns a dictionary mapping each page name to a set of pages it links to.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"', contents)
            pages[filename] = set(links) - {filename}

    # Only include links to pages actually in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    distribution = {}
    num_pages = len(corpus)
    links = corpus[page]

    # If page has no links, treat it as linking to all pages equally
    if not links:
        for p in corpus:
            distribution[p] = 1 / num_pages
        return distribution

    # Probability of picking a random page (the "1 - d" part)
    random_prob = (1 - damping_factor) / num_pages

    # Probability of following a specific link (the "d" part)
    link_prob = damping_factor / len(links)

    for p in corpus:
        # Every page has a baseline probability of being chosen randomly
        distribution[p] = random_prob
        # If the page is linked to by the current page, add the link probability
        if p in links:
            distribution[p] += link_prob

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling n pages
    state space.
    """
    samples_count = {page: 0 for page in corpus}

    # Generate the first sample randomly
    current_page = random.choice(list(corpus.keys()))
    samples_count[current_page] += 1

    # Generate the remaining n-1 samples based on transition model
    for _ in range(n - 1):
        model = transition_model(corpus, current_page, damping_factor)
        pages = list(model.keys())
        weights = list(model.values())

        # Select next page based on calculated probabilities
        current_page = random.choices(pages, weights=weights, k=1)[0]
        samples_count[current_page] += 1

    # Return the proportion of samples for each page
    return {page: count / n for page, count in samples_count.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    values until convergence.
    """
    N = len(corpus)
    # Start with equal rank for all pages
    pageranks = {page: 1 / N for page in corpus}

    # Pre-process: handle pages with no links as if they link to everyone
    new_corpus = {p: (links if links else set(corpus.keys()))
                  for p, links in corpus.items()}

    while True:
        new_ranks = {}
        for page in corpus:
            # Baseline probability: (1 - d) / N
            rank_sum = (1 - damping_factor) / N

            # Link contribution: d * sum(PR(i) / Links(i))
            link_contribution = 0
            for possible_linker in corpus:
                if page in new_corpus[possible_linker]:
                    link_contribution += pageranks[possible_linker] / \
                        len(new_corpus[possible_linker])

            new_ranks[page] = rank_sum + (damping_factor * link_contribution)

        # Check if any PageRank changed by more than 0.001
        differences = [abs(new_ranks[p] - pageranks[p]) for p in corpus]
        if max(differences) < 0.001:
            # Final normalization to ensure sum is exactly 1.0
            total_sum = sum(new_ranks.values())
            return {p: r / total_sum for p, r in new_ranks.items()}

        pageranks = new_ranks.copy()


if __name__ == "__main__":
    main()
