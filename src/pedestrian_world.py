import itertools
import random
from typing import Dict, List, Tuple
from numpy import random

class PedestrianMDP:
    def __init__(self, base_mdp, ped_starts: List[Tuple[int, int]], collision_penalty: float = 100.0):
        self.base_mdp = base_mdp
        self.ped_starts = ped_starts
        self.collision_penalty = collision_penalty

        self.start = tuple([self.base_mdp.start] + self.ped_starts)

        state_lists = [self.base_mdp.states for _ in range(1 + len(self.ped_starts))]
        self.states = list(itertools.product(*state_lists))

        self.N = self.base_mdp.N

        
    def is_goal(self, s) -> bool:
        return self.base_mdp.is_goal(s[0])

    def actions_from(self, s) -> List[str]:
        return self.base_mdp.actions_from(s[0])
    
    def _pedestrian_step_probs(self, ped_pos) -> Dict[Tuple[int, int], float]:
        valid_moves = [ped_pos] 
        for a in ["U", "D", "L", "R"]:
            valid_moves.append(self.base_mdp.move(ped_pos, a))
        
        probs = {}
        for pos in valid_moves:
            probs[pos] = probs.get(pos, 0.0) + (1.0 / len(valid_moves))
        return probs
    

    def transitions(self, s, a) -> Dict[Tuple, float]:
        if self.is_goal(s):
            return {s: 1.0}

        agent_trans = self.base_mdp.transitions(s[0], a)
        
        ped_trans_list = [self._pedestrian_step_probs(p) for p in s[1:]]

        all_trans = [agent_trans] + ped_trans_list
        items_lists = [list(d.items()) for d in all_trans]

        joint_transitions = {}
        for combo in itertools.product(*items_lists):
            next_state = tuple(pos for pos, prob in combo)
            
            joint_prob = 1.0
            for pos, prob in combo:
                joint_prob *= prob
            
            joint_transitions[next_state] = joint_transitions.get(next_state, 0.0) + joint_prob

        return joint_transitions
    


    def cost(self, s, a) -> float:
        exp_cost = 0.0
        for next_s, prob in self.transitions(s, a).items():
            next_agent = next_s[0]
            next_peds = next_s[1:]
            
            step_cost = self.base_mdp.cost_cell(next_agent[0], next_agent[1])
            
            if next_agent in next_peds:
                step_cost += self.collision_penalty
                
            exp_cost += prob * step_cost
            
        return exp_cost

    def move(self, s, a) -> Tuple:
        next_agent = self.base_mdp.move(s[0], a)
        # return tuple([next_agent] + list(s[1:]))
        next_peds = []
        for ped_pos in s[1:]:
            valid_moves = [ped_pos]  
            
            for direction in ["U", "D", "L", "R"]:
                valid_moves.append(self.base_mdp.move(ped_pos, direction))
            
            chosen_ped_move = random.choice(valid_moves)
            next_peds.append(chosen_ped_move)
            
        return tuple([next_agent] + next_peds)

    @property
    def goal(self):
        return self.base_mdp.goal