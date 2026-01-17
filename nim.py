import random
import time


class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has a list of piles, a current player (0 or 1),
        and a winner.
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a list of piles as input
        and returns all of the available actions (i, j) in that state.

        Action (i, j) represents taking 'j' objects from pile 'i'.
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make the move `action` for the current player.
        `action` must be a tuple (i, j).
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already over")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning rate), and an epsilon rate.

        The Q-learning dictionary maps (state, action) pairs to a Q-value (a number).
         - state is a tuple of remaining piles, e.g. (1, 1, 4, 4)
         - action is a tuple (i, j) of pile and number of objects to take
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        # Convert list state to tuple for dictionary key compatibility
        state_tuple = tuple(state)
        return self.q.get((state_tuple, action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estimate of future rewards `future_rewards`.
        """
        state_tuple = tuple(state)
        # New value estimate = current reward + predicted future reward
        new_value_estimate = reward + future_rewards

        # Q-learning formula
        self.q[(state_tuple, action)] = old_q + \
            self.alpha * (new_value_estimate - old_q)

    def best_future_reward(self, state):
        """
        Return the maximum Q-value for any available action in `state`.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0

        # Find the highest Q-value among all possible actions in this state
        best_reward = 0
        for action in actions:
            q_val = self.get_q_value(state, action)
            if q_val > best_reward or best_reward == 0:
                best_reward = q_val

        return best_reward

    def choose_action(self, state, epsilon=True):
        """
        Each element in the `state` list is the number of objects in that pile.
        Return the best action (i, j) for the state.

        If `epsilon` is True, choose a random action with probability `self.epsilon`.
        Otherwise, choose the best action with the highest Q-value.
        """
        actions = list(Nim.available_actions(state))

        # Epsilon-Greedy: Explore (random) vs Exploit (best)
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        # Find the best action (exploitation)
        best_action = None
        max_q = -float('inf')

        for action in actions:
            q_val = self.get_q_value(state, action)
            if q_val > max_q:
                max_q = q_val
                best_action = action

        return best_action


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """
    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:
            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                # For winner, receive reward 1
                player.update(state, action, new_state, 1)
                # For loser, receive reward -1
                player.update(
                    last[game.winner]["state"],
                    last[game.winner]["action"],
                    new_state,
                    -1
                )
                break

            # If game is ongoing, update Q values with 0 reward
            elif last[game.other_player(game.player)]["state"] is not None:
                player.update(
                    last[game.other_player(game.player)]["state"],
                    last[game.other_player(game.player)]["action"],
                    new_state,
                    0
                )

    print("Done training")
    return player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player goes first or second.
    """
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    # Game loop
    while True:
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Get available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Compute next move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")
            action = (pile, count)
        else:
            print("AI's Turn")
            action = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {action[1]} from pile {action[0]}.")

        # Make move
        game.move(action)

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
