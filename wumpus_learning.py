import random
import math
import tensorflow as tf
import time

class WumpusBoard:
    '''
    Set up for square boards only
    '''
    north_orientation = 'N'
    south_orientation = 'S'
    east_orientation = 'E'
    west_orientation = 'W'

    go_forward = 0
    turn_left = 1
    turn_right = 2
    grab = 3
    shoot = 4
    no_op = 5

    go_forward_successful = 0
    go_forward_slipleft = 1
    go_forward_slipright = 2

    def __init__(self, board_size=16, agent_starting_position=12, agent_starting_orientation=east_orientation, number_of_pits=2, number_of_wumpus=1, go_forward_success=0.99):
        '''
        agent_position is assumed indexed from zero
        '''
        self.agent_has_arrow = True
        self.agent_has_gold = False
        self.number_of_pits = number_of_pits
        self.number_of_wumpus = number_of_wumpus
        self.go_forward_success = go_forward_success

        self.wumpus_board = [[False for j in xrange(2 + number_of_pits + number_of_wumpus)] for i in xrange(board_size)]

        # set agent position
        self.wumpus_board[agent_starting_position][0] = agent_starting_orientation

        # set gold
        self.wumpus_board[random.randint(0, board_size - 1)][1] = True

        # set pits
        pit_in_position = [False for i in xrange(board_size)]
        for i in xrange(number_of_pits):
            pit_location = random.randint(0, board_size - 1)

            while pit_location == agent_starting_position or pit_in_position[pit_location] == True:
                pit_location = random.randint(0, board_size - 1)

            self.wumpus_board[pit_location][2 + i] = True
            pit_in_position[pit_location] = True

        # set wumpus
        for i in xrange(number_of_wumpus):
            wumpus_location = random.randint(0, board_size - 1)

            while wumpus_location == agent_starting_position:
                wumpus_location = random.randint(0, board_size - 1)

            self.wumpus_board[wumpus_location][2 + number_of_pits + i] = True

        self.update_observation()

    def update_observation(self, bump=False, scream=False):
        self.current_observation = {}
        pits_near, wumpus_near = self.is_agent_adjacent_to_danger()
        on_gold = self.is_agent_on_gold()

        self.current_observation['breeze'] = pits_near
        self.current_observation['stench'] = wumpus_near
        self.current_observation['glitter'] = on_gold
        self.current_observation['bump'] = bump
        self.current_observation['scream'] = scream

    def get_current_observation(self):
        return self.current_observation

    def is_agent_adjacent_to_danger(self):
        board_dimension = int(math.sqrt(len(self.wumpus_board)))
        possible_danger_locations = []
        wumpus_near = False
        pits_near = False

        agent_position = 0
        for i in xrange(len(self.wumpus_board)):
            if self.wumpus_board[i][0] != False:
                agent_position = i

        agent_row_index = agent_position / board_dimension
        agent_column_index = agent_position % board_dimension

        # calculate cross positions
        # above
        if agent_row_index - 1 >= 0:
            possible_danger_locations.append((agent_row_index - 1)*board_dimension + agent_column_index)

        # below
        if agent_row_index + 1 < board_dimension:
            possible_danger_locations.append((agent_row_index + 1)*board_dimension + agent_column_index)
        # right
        if agent_column_index + 1 < board_dimension:
            possible_danger_locations.append((agent_row_index)*board_dimension + agent_column_index + 1)

        # left
        if agent_column_index - 1 >= 0:
            possible_danger_locations.append((agent_row_index)*board_dimension + agent_column_index - 1)

        # current location
        possible_danger_locations.append(agent_position)

        for possible_danger_location in possible_danger_locations:
            # check for pits
            for j in xrange(2, 2 + self.number_of_pits):
                if self.wumpus_board[possible_danger_location][j] == True:
                    pits_near = True


            # check for wumpus
            for j in xrange(2 + self.number_of_pits, 2 + self.number_of_pits + self.number_of_wumpus):
                if self.wumpus_board[possible_danger_location][j] == True:
                    wumpus_near = True

        return pits_near, wumpus_near

    def is_agent_on_gold(self):
        on_gold = False

        for i in xrange(len(self.wumpus_board)):
            if self.wumpus_board[i][0] != False:
                if self.wumpus_board[i][1] == True:
                    on_gold = True
                break

        return on_gold

    def print_board(self):
        board_dimension = int(math.sqrt(len(self.wumpus_board)))
        board_space_str = ''

        for i in xrange(len(self.wumpus_board)):
            if i != 0 and i % board_dimension == 0:
                print board_space_str
                board_space_str = ''

            for j in xrange(len(self.wumpus_board[i])):

                if j == 0:
                    if self.wumpus_board[i][j] == False:
                        board_space_str += '_'
                    else:
                        board_space_str += self.wumpus_board[i][j]
                elif j == 1:
                    if self.wumpus_board[i][j] == False:
                        board_space_str += '_'
                    else:
                        board_space_str += 'G'
                elif j > 1 and j <= 1 + self.number_of_pits:
                    if self.wumpus_board[i][j] == False:
                        board_space_str += '_'
                    else:
                        board_space_str += 'P'
                elif j > 1 + self.number_of_pits:
                    if self.wumpus_board[i][j] == False:
                        board_space_str += '_'
                    else:
                        board_space_str += 'W'
            board_space_str += '|'

        print board_space_str

    def nondeterministic_goforward_result(self):
        random_number = random.random()

        if random_number < self.go_forward_success:
            return self.go_forward_successful
        elif random_number < self.go_forward_success + (1.0 - self.go_forward_success)/2:
            return self.go_forward_slipright
        else:
            return self.go_forward_slipleft

    def perform_agent_action(self, action):
        board_dimension = int(math.sqrt(len(self.wumpus_board)))
        agent_position = 0
        gold_position = 0
        agent_orientation = self.north_orientation
        for i in xrange(len(self.wumpus_board)):
            if self.wumpus_board[i][0] != False:
                agent_position = i
                agent_orientation = self.wumpus_board[i][0]
            if self.wumpus_board[i][1] != False:
                gold_position = i

        agent_row_index = agent_position / board_dimension
        agent_column_index = agent_position % board_dimension

        if action == self.go_forward:
            bump = False
            forward_result = self.nondeterministic_goforward_result()

            if agent_orientation == self.north_orientation:
                if forward_result == self.go_forward_successful:
                    if agent_row_index - 1 >= 0:
                        self.wumpus_board[(agent_row_index - 1)*board_dimension + agent_column_index][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                elif forward_result == self.go_forward_slipleft:
                    if agent_column_index - 1 >= 0:
                        self.wumpus_board[agent_row_index*board_dimension + agent_column_index - 1][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                else:
                    if agent_column_index + 1 < board_dimension:
                        self.wumpus_board[agent_row_index*board_dimension + agent_column_index + 1][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True

            elif agent_orientation == self.south_orientation:
                if forward_result == self.go_forward_successful:
                    if agent_row_index + 1 < board_dimension:
                        self.wumpus_board[(agent_row_index + 1)*board_dimension + agent_column_index][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                elif forward_result == self.go_forward_slipleft:
                    if agent_column_index + 1 < board_dimension:
                        self.wumpus_board[agent_row_index*board_dimension + agent_column_index + 1][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                else:
                    if agent_column_index - 1 >= 0:
                        self.wumpus_board[agent_row_index*board_dimension + agent_column_index - 1][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
            elif agent_orientation == self.east_orientation:
                if forward_result == self.go_forward_successful:
                    if agent_column_index + 1 < board_dimension:
                        self.wumpus_board[agent_row_index*board_dimension + agent_column_index + 1][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                elif forward_result == self.go_forward_slipleft:
                    if agent_row_index - 1 >= 0:
                        self.wumpus_board[(agent_row_index - 1)*board_dimension + agent_column_index][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                else:
                    if agent_row_index + 1 < board_dimension:
                        self.wumpus_board[(agent_row_index + 1)*board_dimension + agent_column_index][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
            elif agent_orientation == self.west_orientation:
                if forward_result == self.go_forward_successful:
                    if agent_column_index - 1 >= 0:
                        self.wumpus_board[agent_row_index*board_dimension + agent_column_index - 1][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                elif forward_result == self.go_forward_slipleft:
                    if agent_row_index + 1 < board_dimension:
                        self.wumpus_board[(agent_row_index + 1)*board_dimension + agent_column_index][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
                else:
                    if agent_row_index - 1 >= 0:
                        self.wumpus_board[(agent_row_index - 1)*board_dimension + agent_column_index][0] = agent_orientation
                        self.wumpus_board[agent_position][0] = False
                    else:
                        bump = True
            self.update_observation(bump=bump)

        elif action == self.turn_right:
            if agent_orientation == self.north_orientation:
                self.wumpus_board[i][0] = self.east_orientation
            elif agent_orientation == self.south_orientation:
                self.wumpus_board[i][0] = self.west_orientation
            elif agent_orientation == self.east_orientation:
                self.wumpus_board[i][0] = self.south_orientation
            elif agent_orientation == self.west_orientation:
                self.wumpus_board[i][0] = self.north_orientation
        elif action == self.turn_left:
            if agent_orientation == self.north_orientation:
                self.wumpus_board[i][0] = self.west_orientation
            elif agent_orientation == self.south_orientation:
                self.wumpus_board[i][0] = self.east_orientation
            elif agent_orientation == self.east_orientation:
                self.wumpus_board[i][0] = self.north_orientation
            elif agent_orientation == self.west_orientation:
                self.wumpus_board[i][0] = self.south_orientation
        elif action == self.grab:
            if agent_position == gold_position:
                self.agent_has_gold = True
                self.wumpus_board[i][1] = False
        elif action == self.shoot:
            if self.agent_has_arrow == True:
                agent_row_index = agent_position / board_dimension
                agent_column_index = agent_position % board_dimension

                if agent_orientation == self.north_orientation:
                    forward_positions = []
                    temp_row_index = agent_row_index - 1

                    while temp_row_index >= 0:
                        forward_positions.append(temp_row_index*board_dimension + agent_column_index)
                        temp_row_index -= 1


                elif agent_orientation == self.south_orientation:
                    forward_positions = []
                    temp_row_index = agent_row_index + 1

                    while temp_row_index < board_dimension:
                        forward_positions.append(temp_row_index*board_dimension + agent_column_index)
                        temp_row_index += 1

                elif agent_orientation == self.east_orientation:
                    forward_positions = []
                    temp_column_index = agent_column_index + 1

                    while temp_column_index < board_dimension:
                        forward_positions.append(agent_row_index*board_dimension + temp_column_index)
                        temp_column_index += 1


                elif agent_orientation == self.west_orientation:
                    forward_positions = []
                    temp_column_index = agent_column_index - 1

                    while temp_column_index >= 0:
                        forward_positions.append(agent_row_index*board_dimension + temp_column_index)
                        temp_column_index -= 1

                scream = False
                for position in forward_positions:
                    for i in xrange(2 + self.number_of_pits, 2 + self.number_of_pits + self.number_of_wumpus):
                        if self.wumpus_board[position][i] == True:
                            self.wumpus_board[position][i] = False
                            scream = True

                self.update_observation(scream=scream)
                self.agent_has_arrow = False

        elif action == self.no_op:
            pass
        else:
            raise Exception('Not valid action')

    def does_agent_has_arrow(self):
        return self.agent_has_arrow

    def does_agent_has_gold(self):
        return self.agent_has_gold

    def has_agent_died(self):
        for i in xrange(len(self.wumpus_board)):
            for j in xrange(2, 2 + self.number_of_pits + self.number_of_wumpus):
                if self.wumpus_board[i][0] != False and self.wumpus_board[i][j] == True:
                    return True

        return False

class Reward:
    death = -1000
    pickup_gold = 1000
    action_cost = -1
    shoot_cost = -10

def main():
    #inputs
    batch_size = 32
    past_history = 10
    number_of_observations = 5
    discount_factor = 0.75
    hidden_layer_size = 75
    number_of_training_trials = 100
    games_per_training_trial = 1000
    games_per_testing_trial = 1000
    turns_per_game = 100
    replay_memory_capacity = 1000000
    training_report_interval = 50

    action_mapping = {}
    action_mapping[WumpusBoard.go_forward] = -1.0
    action_mapping[WumpusBoard.turn_left] = -0.66
    action_mapping[WumpusBoard.turn_right] = -0.33
    action_mapping[WumpusBoard.shoot] = 0.33
    action_mapping[WumpusBoard.grab] = 0.66
    action_mapping[WumpusBoard.no_op] = 1.0

    action_list = [WumpusBoard.go_forward, WumpusBoard.turn_right, WumpusBoard.turn_left, WumpusBoard.shoot, WumpusBoard.grab, WumpusBoard.no_op]

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, (past_history - 1)*(number_of_observations + 1) + number_of_observations + 1))
        tf_train_target = tf.placeholder(tf.float32, shape=(batch_size, 1))

        tf_test_dataset = tf.placeholder(tf.float32, shape=(1, (past_history - 1)*(number_of_observations + 1) + number_of_observations + 1))

        # Variables.
        weights1 = tf.Variable(tf.truncated_normal([(past_history - 1)*(number_of_observations + 1) + number_of_observations + 1, hidden_layer_size]))
        biases1 = tf.Variable(tf.zeros([hidden_layer_size]))

        weights2 = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size]))
        biases2 = tf.Variable(tf.zeros([hidden_layer_size]))


        weights3 = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size]))
        biases3 = tf.Variable(tf.zeros([hidden_layer_size]))


        weights4 = tf.Variable(tf.truncated_normal([hidden_layer_size, 1]))
        biases4 = tf.Variable(tf.zeros([1]))

        # Training computation.
        logit1 = tf.matmul(tf_train_dataset, weights1) + biases1
        relu1 = tf.nn.relu(logit1)
        logit2 = tf.matmul(relu1, weights2) + biases2
        relu2 = tf.nn.relu(logit2)
        logit3 = tf.matmul(relu2, weights3) + biases3
        relu3 = tf.nn.relu(logit3)
        training_output = tf.matmul(relu3, weights4) + biases4

        loss = tf.reduce_mean(tf.square(tf_train_target - training_output))
        optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)

        # Testing computations
        testing_logit1 = tf.matmul(tf_test_dataset, weights1) + biases1
        testing_relu1 = tf.nn.relu(testing_logit1)
        testing_logit2 = tf.matmul(testing_relu1, weights2) + biases2
        testing_relu2 = tf.nn.relu(testing_logit2)
        testing_logit3 = tf.matmul(testing_relu2, weights3) + biases3
        testing_relu3 = tf.nn.relu(testing_logit3)
        test_output = tf.matmul(testing_relu3, weights4) + biases4

    with tf.Session(graph=graph, config=tf.ConfigProto(intra_op_parallelism_threads=4)) as session:
        tf.initialize_all_variables().run()
        replay_bank = []
        epsilon = 1.0

        for epoch in xrange(number_of_training_trials):
            print "[%f] On Epoch %d out of %d" % (time.time(),epoch + 1, number_of_training_trials)
            print "[%f] Training Phase" % (time.time(),)

            if epoch > 0 and epsilon > 0.1:
                epsilon -= 0.1

            #training phase
            losses = []
            for game in xrange(games_per_training_trial):
                observation_sequence = [0.0 for i in xrange(number_of_observations * past_history + past_history)]
                board = WumpusBoard()
                at_terminating_state = False

                for turn in xrange(1, turns_per_game + 1):
                    turn_score = 0

                    observation = board.get_current_observation()

                    for i in xrange(number_of_observations):
                        observation_sequence.pop(0)
                    observation_sequence.pop(0)

                    if observation['breeze']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['stench']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['glitter']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['bump']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['scream']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)

                    # explore or follow network
                    if random.random() < epsilon:
                        action_chosen = random.sample(action_list, 1)[0]
                    else:
                        action_qvalues = []
                        for action, action_encoding in action_mapping.iteritems():
                            data_input = [observation_sequence + [action_encoding]]
                            feed_dict = {tf_test_dataset : data_input}

                            qvalue = session.run([test_output], feed_dict=feed_dict)
                            action_qvalues.append((action, qvalue))
                        action_chosen, _ = max(action_qvalues, key=lambda x:x[1])

                    if action_chosen == WumpusBoard.shoot and board.does_agent_has_arrow():
                        turn_score += Reward.shoot_cost

                    board.perform_agent_action(action_chosen)


                    if board.has_agent_died():
                        turn_score += Reward.death
                        at_terminating_state = True
                    elif board.does_agent_has_gold():
                        turn_score += Reward.pickup_gold
                        at_terminating_state = True

                    if action_chosen == WumpusBoard.no_op:
                        turn_score += 0
                    else:
                        turn_score += Reward.action_cost

                    if len(replay_bank) > replay_memory_capacity:
                        replay_bank.pop(0)

                    if at_terminating_state:
                        replay_bank.append((observation_sequence[:], action_mapping[action_chosen], turn_score, True))
                    else:
                        next_observation_sequence = observation_sequence[number_of_observations + 1:] + [action_mapping[action_chosen]]
                        next_observation = board.get_current_observation()

                        if next_observation['breeze']:
                            next_observation_sequence.append(1.0)
                        else:
                            next_observation_sequence.append(-1.0)
                        if next_observation['stench']:
                            next_observation_sequence.append(1.0)
                        else:
                            next_observation_sequence.append(-1.0)
                        if next_observation['glitter']:
                            next_observation_sequence.append(1.0)
                        else:
                            next_observation_sequence.append(-1.0)
                        if next_observation['bump']:
                            next_observation_sequence.append(1.0)
                        else:
                            next_observation_sequence.append(-1.0)
                        if next_observation['scream']:
                            next_observation_sequence.append(1.0)
                        else:
                            next_observation_sequence.append(-1.0)

                        replay_bank.append((observation_sequence[:], action_mapping[action_chosen], turn_score, next_observation_sequence))

                    observation_sequence.append(action_mapping[action_chosen])

                    # update graph
                    if len(replay_bank) >= batch_size:
                        batch_data = random.sample(replay_bank, batch_size)
                        training_data = []
                        training_target = []

                        for batch in batch_data:
                            training_data.append(batch[0] + [batch[1]])

                            if batch[3] == True:
                                training_target.append([batch[2]])
                            else:
                                action_qvalues = []
                                for action, action_encoding in action_mapping.iteritems():
                                    data_input = [batch[3] + [action_encoding]]
                                    feed_dict = {tf_test_dataset : data_input}

                                    qvalue = session.run([test_output], feed_dict=feed_dict)
                                    action_qvalues.append((action, qvalue))
                                best_action, best_qvalue = max(action_qvalues, key=lambda x:x[1])
                                training_target.append([batch[2] + discount_factor * best_qvalue[0][0,0]])

                        feed_dict = {tf_train_dataset:training_data, tf_train_target:training_target}
                        _, l, predictions = session.run([optimizer, loss, training_output], feed_dict=feed_dict)
                        losses.append(l)

                    # end game if terminating:
                    if at_terminating_state:
                        break

                if (game + 1) % training_report_interval == 0 or (game + 1) == games_per_training_trial:
                    print '[%f] Training Progress: %d out of the %d training games completed, average loss for this interval was %f' % (time.time(), game + 1, games_per_training_trial, sum(losses)/float(len(losses)))
                    losses = []

            print "[%f] Testing Phase" % (time.time(),)
            #testing phase
            testing_scores = []
            for game in xrange(games_per_testing_trial):
                observation_sequence = [0.0 for i in xrange(number_of_observations * past_history + past_history)]
                board = WumpusBoard()
                game_score = 0

                for turn in xrange(1, turns_per_game + 1):
                    if board.has_agent_died():
                        game_score += Reward.death
                        break
                    elif board.does_agent_has_gold():
                        game_score += Reward.pickup_gold
                        break

                    observation = board.get_current_observation()

                    for i in xrange(number_of_observations):
                        observation_sequence.pop(0)
                    observation_sequence.pop(0)

                    if observation['breeze']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['stench']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['glitter']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['bump']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)
                    if observation['scream']:
                        observation_sequence.append(1.0)
                    else:
                        observation_sequence.append(-1.0)

                    action_qvalues = []
                    for action, action_encoding in action_mapping.iteritems():
                        data_input = [observation_sequence + [action_encoding]]
                        feed_dict = {tf_test_dataset : data_input}

                        qvalue = session.run([test_output], feed_dict=feed_dict)
                        action_qvalues.append((action, qvalue))
                    best_action, _ = max(action_qvalues, key=lambda x:x[1])

                    if best_action == WumpusBoard.shoot and board.does_agent_has_arrow():
                        game_score += Reward.shoot_cost

                    board.perform_agent_action(best_action)
                    observation_sequence.append(action_mapping[best_action])

                    if best_action == WumpusBoard.no_op:
                        game_score += 0
                    else:
                        game_score += Reward.action_cost

                testing_scores.append(game_score)
            print "[%f] Average score for %d is %f" % (time.time(), games_per_testing_trial, sum(testing_scores) / float(len(testing_scores)))
            print "***********************"
            print ""

main()
