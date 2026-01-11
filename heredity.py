import csv
import itertools
import sys

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }
    names = set(people)
    for have_gene in powerset(names):
        for one_gene in powerset(have_gene):
            two_genes = have_gene - one_gene
            for have_trait in powerset(names):
                ground_truth = joint_probability(
                    people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes,
                       have_trait, ground_truth)
    normalize(probabilities)
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    s = list(s)
    return [
        set(subset) for i in range(len(s) + 1)
        for subset in itertools.combinations(s, i)
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    joint_p = 1
    for person in people:
        p_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        p_trait = person in have_trait

        # Gene probability
        if not people[person]['mother']:
            # No parents: use unconditional probability
            gene_p = PROBS["gene"][p_genes]
        else:
            # Has parents: calculate based on inheritance
            mother = people[person]['mother']
            father = people[person]['father']
            probabilities = {}

            for parent in [mother, father]:
                parent_genes = 2 if parent in two_genes else 1 if parent in one_gene else 0
                if parent_genes == 0:
                    probabilities[parent] = PROBS["mutation"]
                elif parent_genes == 1:
                    probabilities[parent] = 0.5
                else:
                    probabilities[parent] = 1 - PROBS["mutation"]

            if p_genes == 2:
                gene_p = probabilities[mother] * probabilities[father]
            elif p_genes == 1:
                gene_p = (probabilities[mother] * (1 - probabilities[father])) + \
                         ((1 - probabilities[mother]) * probabilities[father])
            else:
                gene_p = (1 - probabilities[mother]) * \
                    (1 - probabilities[father])

        # Trait probability
        trait_p = PROBS["trait"][p_genes][p_trait]
        joint_p *= gene_p * trait_p

    return joint_p


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        p_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        p_trait = person in have_trait
        probabilities[person]["gene"][p_genes] += p
        probabilities[person]["trait"][p_trait] += p


def normalize(probabilities):
    for person in probabilities:
        for field in ["gene", "trait"]:
            total = sum(probabilities[person][field].values())
            for val in probabilities[person][field]:
                probabilities[person][field][val] /= total


if __name__ == "__main__":
    main()
