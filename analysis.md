# Analysis

## Layer 6, Head 10

This attention head appears to specialize in connecting pronouns to the verbs and adverbs that immediately follow them. The head consistently shows that pronouns (like "she", "he", "they") attend strongly to action-describing words that come next in the sentence. This suggests the head is learning to link the agent of an action (the pronoun) to the action itself, which is crucial for understanding who is doing what in a sentence.

Example Sentences:

- "she quickly walked to the [MASK]" - Here, "she" attends strongly to "quickly" (0.42 probability), which is the adverb modifying her action.
- "he runs very fast to the [MASK]" - The pronoun "he" attends strongly to "runs" (0.79 probability), directly linking the agent to the verb.
- "they jump over the [MASK]" - Similarly, "they" attends to "jump" (0.60 probability), establishing the agent-action relationship.

## Layer 7, Head 6

This attention head specializes in the grammatical relationship between prepositions and their objects. Prepositions consistently attend to the nouns that follow them with very high attention weights (often 0.88-0.99), revealing that this head has learned to identify the fundamental prepositional phrase structure. This is one of the most reliable syntactic relationships in language, and the model has clearly dedicated a specific head to capturing it.

Example Sentences:

- "the cat on the mat sat [MASK]" - The preposition "on" attends almost exclusively to "mat" (0.94 probability), showing perfect syntactic binding of the preposition to its object.
- "the dog in the house ran [MASK]" - The preposition "in" attends with 0.22 probability to "house", while "the" before "house" shows 0.56 probability to "house", establishing the prepositional phrase structure.
- "the ball under the table was [MASK]" - The preposition "under" shows strong focus on "table", with the determiner "the" before "table" attending to "table" with 0.43 probability, again revealing the prep-object binding pattern.
