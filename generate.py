import sys
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """Initialize the creator with a crossword puzzle."""
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    # --- THE FUNCTIONS YOU IMPLEMENTED GO HERE (Inside the class) ---

    def enforce_node_consistency(self):
        for var in self.domains:
            for word in set(self.domains[var]):
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        revised = False
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False
        i, j = overlap
        for x_word in set(self.domains[x]):
            if not any(x_word[i] == y_word[j] for y_word in self.domains[y]):
                self.domains[x].remove(x_word)
                revised = True
        return revised

    def ac3(self, arcs=None):
        if arcs is None:
            queue = [(v1, v2)
                     for v1 in self.domains for v2 in self.crossword.neighbors(v1)]
        else:
            queue = list(arcs)
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(x):
                    if neighbor != y:
                        queue.append((neighbor, x))
        return True

    def assignment_complete(self, assignment):
        return len(assignment) == len(self.crossword.variables)

    def consistent(self, assignment):
        words = list(assignment.values())
        if len(words) != len(set(words)):
            return False
        for var, word in assignment.items():
            if len(word) != var.length:
                return False
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if word[i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        def count_eliminations(val):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    for n_word in self.domains[neighbor]:
                        if val[i] != n_word[j]:
                            count += 1
            return count
        return sorted(self.domains[var], key=count_eliminations)

    def select_unassigned_variable(self, assignment):
        unassigned = [
            v for v in self.crossword.variables if v not in assignment]
        return min(unassigned, key=lambda v: (len(self.domains[v]), -len(self.crossword.neighbors(v))))

    def backtrack(self, assignment):
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result:
                    return result
        return None

    def solve(self):
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def print(self, assignment):
        """Print crossword assignment to the terminal."""
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def letter_grid(self, assignment):
        """Return 2D array of characters for a given assignment."""
        letters = [[None for _ in range(self.crossword.width)]
                   for _ in range(self.crossword.height)]
        for var, word in assignment.items():
            row, col = var.i, var.j
            for k in range(len(word)):
                letters[row + (k if var.direction == Variable.DOWN else 0)][col +
                                                                            (k if var.direction == Variable.ACROSS else 0)] = word[k]
        return letters

    def save(self, assignment, filename):
        """Save crossword assignment to an image file."""
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)
        img = Image.new("RGB", (self.crossword.width * cell_size,
                        self.crossword.height * cell_size), "black")
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [(j * cell_size + cell_border, i * cell_size + cell_border),
                        ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox(
                            (0, 0), letters[i][j], font=font)
                        draw.text((rect[0][0] + (interior_size - w) / 2, rect[0][1] + (
                            interior_size - h) / 2 - 10), letters[i][j], fill="black", font=font)
        img.save(filename)


def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
