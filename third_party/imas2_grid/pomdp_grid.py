from itertools import product

# self.grid_world = Environment()

class POMDP:

    def __init__(self, grid_world):
        # Import the grid world environment
        self.grid_world = grid_world
        # Define states
        # Here tau = 0 (L(s) = !t) means nominal agent and tau = 1 (L(s) = t) means adversary
        self.states = [(gr_st, tau) for gr_st, tau in product(self.grid_world.gr_states, [0, 1])]
        # self.state_indices = list(range(len(self.states)))
        # self.state_size = len(self.states)
        # Define initial state
        self.initial_states = [(init_st, tau) for init_st, tau in product(self.grid_world.gr_initial_states, [0, 1])]
        # self.initial_state_idx = self.states.index(self.initial_state)
        # Define perception actions
        self.actions = ['NW', 'NE', 'SW', 'SE']
        self.action_size = len(self.actions)
        self.action_indices = list(range(len(self.actions)))
        # transition probability dictionary
        self.next_supp = self.get_next_supp_with_action()
        self.transition = self.get_transition()
        self.check_the_transition()
        # Define UAV with sensors
        self.obs_noise = self.grid_world.obs_noise  # the noise of sensors
        # Define observations
        self.observations = [(gr_st, sensor_st) for gr_st, sensor_st in
                       product((self.grid_world.gr_states + ['n']), self.grid_world.sensors)]
        self.obs_dict_list = [self.get_observation_dictionary(sn) for sn in range(len(self.grid_world.sensors))]
        self.emiss_list = [self.get_emission_function(sn) for sn in range(len(self.grid_world.sensors))]
        self.check_emission_function()
        # Define the atomic propositions
        self.atom_prop = ['t']  # t indicates adversary
        # Define the labeling function
        self.label_func = self.get_label_function()

    def get_next_supp_with_action(self):
        next_supp = {}
        for st in self.states:
            next_supp[st] = {}
            for act in self.actions:
                if st[1] == 0:
                    gr_next_supp = list(self.grid_world.transition_n[st[0]].keys())
                elif st[1] == 1:
                    gr_next_supp = list(self.grid_world.transition_a[st[0]].keys())
                else:
                    raise ValueError('Invalid type value.')
                next_supp[st][act] = [(gr_st, st[1]) for gr_st in gr_next_supp]
        return next_supp

    def get_transition(self):
        trans = {}
        for st in self.states:
            trans[st] = {}
            for act in self.actions:
                trans[st][act] = {}
                for st_prime in self.next_supp[st][act]:
                    if st[1] == 0:
                        trans[st][act][st_prime] = self.grid_world.transition_n[st[0]][st_prime[0]]
                    elif st[1] == 1:
                        trans[st][act][st_prime] = self.grid_world.transition_a[st[0]][st_prime[0]]
                    else:
                        raise ValueError('Invalid type value.')
        return trans

    def check_the_transition(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                for st_prime in self.next_supp[st][act]:
                    prob += self.transition[st][act][st_prime]
                if abs(prob - 1) > 0.01:
                    print("The transition is invalid.", self.transition[st][act])
        print("The check of transition function is done (POMDP).")
        return 0

    def in_sensor_range(self, st, act, sensor_num):
        sensor_pos = self.grid_world.sensors[sensor_num]
        gr_pos = st[0]
        if act == 'NW':
            sensor_range_to_add = [(0, 0), (-1, 0), (-1, 1), (0, 1)]
            # sensor_range_to_add = [(0, 0), (-1, 0), (-1, 1), (0, 1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2)]
        elif act == 'NE':
            sensor_range_to_add = [(0, 0), (1, 0), (1, 1), (0, 1)]
            # sensor_range_to_add = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
        elif act == 'SW':
            sensor_range_to_add = [(0, 0), (-1, 0), (-1, -1), (0, -1)]
            # sensor_range_to_add = [(0, 0), (-1, 0), (-1, -1), (0, -1), (-2, 0), (-2, -1), (-2, -2), (-1, -2), (0, -2)]
        elif act == 'SE':
            sensor_range_to_add = [(0, 0), (1, 0), (1, -1), (0, -1)]
            # sensor_range_to_add = [(0, 0), (1, 0), (1, -1), (0, -1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2)]
        else:
            raise ValueError('Invalid perception action.')
        sensor_range = [(sensor_pos[0] + to_add[0], sensor_pos[1] + to_add[1]) for to_add in sensor_range_to_add]
        if gr_pos in sensor_range:
            return True
        else:
            return False

    def get_observation_dictionary(self, sensor_num):
        obs_dict = {}
        sensor_pos = self.grid_world.sensors[sensor_num]
        for st in self.states:
            obs_dict[st] = {}
            for act in self.actions:
                if self.in_sensor_range(st, act, sensor_num):
                    obs_dict[st][act] = [(st[0], sensor_pos), ('n', sensor_pos)]
                else:
                    obs_dict[st][act] = [('n', sensor_pos)]
        return obs_dict

    def get_emission_function(self, sensor_num):
        emiss = {}
        obs_dict = self.obs_dict_list[sensor_num]
        for st in self.states:
            emiss[st] = {}
            for act in self.actions:
                emiss[st][act] = {}
                for obs in self.observations:
                    if obs in obs_dict[st][act]:
                        if self.in_sensor_range(st, act, sensor_num):
                            if st[0] == obs[0]:
                                emiss[st][act][obs] = 1 - self.obs_noise
                            else:
                                emiss[st][act][obs] = self.obs_noise
                        else:
                            emiss[st][act][obs] = 1
                    else:
                        emiss[st][act][obs] = 0
        return emiss

    def check_emission_function(self):
        for emiss in self.emiss_list:
            for st in self.states:
                for act in self.actions:
                    prob = 0
                    for obs in self.observations:
                        prob += emiss[st][act][obs]
                    if abs(prob - 1) > 0.01:
                        print(f"The emission {self.emiss_list.index(emiss)} is invalid.", emiss[st][act])
        print("The check of emission function is done (POMDP).")
        return 0

    def get_label_function(self):
        lab = {}
        for st in self.states:
            lab[st] = []
            if st[1] == 1:
                lab[st] += self.atom_prop[0]
        return lab
