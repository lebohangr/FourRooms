import numpy
import random
from FourRooms import FourRooms
import sys

def main():
    # Create FourRooms Object
    if len(sys.argv) > 1:
        if sys.argv[1] == "-stochastic":
            fourRoomsObj = FourRooms('rgb', stochastic=True)
    else:
        fourRoomsObj = FourRooms('rgb')

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    action_space_size = 4
    state_space_size = 169

    q_table = numpy.zeros((state_space_size, action_space_size))

    num_episodes = 10000
    max_steps_per_episode = 100
    learning_rate = 0.01
    discount_rate = 0.9

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    rewards_all_episodes = []
    # Q-Learning algorithm
    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()
        location = fourRoomsObj.getPosition()
        state = 4 * location[1] + location[0]
        rewards_current_episode = 0
        visitedPackages = set()

        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)

            if exploration_rate_threshold > exploration_rate:
                action = numpy.argmax(q_table[state, :])
            else:
                action = random.randint(0, 3)

            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

            newPos = 4 * newPos[1] + newPos[0]


            if gridType > 0 and packagesRemaining == 2 and newPos not in visitedPackages:
                reward = 1
                visitedPackages.add(newPos)
            if gridType > 0 and packagesRemaining == 1 and newPos not in visitedPackages:
                reward = 20
                visitedPackages.add(newPos)
            if (gridType > 0 and packagesRemaining == 0 and newPos not in visitedPackages) or isTerminal:
                reward = 100
                visitedPackages.add(newPos)
            if gridType == 0:
                reward = 0
                # visitedPackages.add((newPos, action))
            if gridType < 0:
                reward = 0
            # # print(q_table)

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

    rewards_per_thousand_episodes = numpy.split(numpy.array(rewards_all_episodes), num_episodes / 1000)
    count = 1000
    print("********Average reward per thousand episodes*********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ":", str(sum(r / 100)))
        count += 1000

    # Print updated Q-table
    # print("\n\n********Q-table******** \n")
    # print(q_table)

    # Show Path
    fourRoomsObj.showPath(-1)

    # fourRoomsObj.stochastic = True


if __name__ == "__main__":
    main()
