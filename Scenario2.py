import numpy
import random
from FourRooms import FourRooms
import sys
from numpy import set_printoptions

set_printoptions(suppress=True)


def main():
    # Create FourRooms Object
    if len(sys.argv) > 1:
        if sys.argv[1] == "-stochastic":
            fourRoomsObj = FourRooms('multi', stochastic=True)
    else:
        fourRoomsObj = FourRooms('multi')

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    action_space_size = 4
    state_space_size = 169

    q_table = numpy.zeros((state_space_size, action_space_size))

    num_episodes = 10000
    max_steps_per_episode = 100
    learning_rate = 0.9
    discount_rate = 1

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.1
    exploration_decay_rate = 0.9999
    rewards_all_episodes = []
    # Q-Learning algorithm
    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()
        location = fourRoomsObj.getPosition()
        state = 13 * location[1] + location[0]
        rewards_current_episode = 0
        visitedPackages = set()
       # print("starting at: " + "{" + str(location[0]) + "," + str(location[1]) + "}")
        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)

            if exploration_rate_threshold > exploration_rate:
                action = numpy.argmax(q_table[state, :])
            else:
                action = random.randint(0, 3)
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

        #    print("moved to: " + "{" + str(newPos[0]) + "," + str(newPos[1]) + "}")

            newPos = 13 * newPos[1] + newPos[0]

            if gridType > 0 and newPos not in visitedPackages and packagesRemaining == 3:
                reward = 10
            #    visitedPackages.add(newPos)
            elif gridType > 0 and packagesRemaining == 1 and newPos not in visitedPackages:
                reward = 10
             #   visitedPackages.add(newPos)
            elif gridType > 0 and packagesRemaining == 0 and newPos not in visitedPackages:
                reward = 10
                visitedPackages.add(newPos)
            elif gridType == 0:
                reward = -1

                # visitedPackages.add((newPos, action))
            # print(reward)
            # Update Q-table for Q(s,a)
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (
                    reward + discount_rate * numpy.max(q_table[newPos, :]))
            state = newPos
            rewards_current_episode += reward

            if isTerminal:
                break

        print("################################################################\n")


        # Exploration rate decay
        exploration_rate = max(min_exploration_rate, exploration_rate*exploration_decay_rate)
        print(exploration_rate)
        rewards_all_episodes.append(rewards_current_episode)
        # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = numpy.split(numpy.array(rewards_all_episodes), num_episodes / 100)
    count = 100

    print("\n********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r / 100)))
        count += 100

    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
