import numpy
import random
from FourRooms import FourRooms
import sys

def main():
    # Create FourRooms Object
    if len(sys.argv) > 1:
        if sys.argv[1] == "-stochastic":
            fourRoomsObj = FourRooms('simple', stochastic=True)
    else:
        fourRoomsObj = FourRooms('simple')


    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    action_space_size = 4
    state_space_size = 169

    q_table = numpy.zeros((state_space_size, action_space_size))

    num_episodes = 500
    max_steps_per_episode = 100
    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01

    rewards_all_episodes = []
    # Q-Learning algorithm
    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()
        location = fourRoomsObj.getPosition()
        state = 13 * location[1] + location[0]
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = numpy.argmax(q_table[state, :])
            else:
                action = random.randint(0, 3)

            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

            newPos = 13 * newPos[1] + newPos[0]

            if gridType > 0:
                reward = 100
            elif gridType == 0:
                reward = -10


            # Update Q-table for Q(s,a)
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (
                    reward + discount_rate * numpy.max(q_table[newPos, :]))
            state = newPos
            rewards_current_episode += reward
            if isTerminal:
                break

        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * numpy.exp(-exploration_decay_rate * episode)
        rewards_all_episodes.append(rewards_current_episode)

    # Show Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
