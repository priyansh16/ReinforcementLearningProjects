import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                agent.adversary = False
                agent.color = np.array([0.25, 0.25, 0.75])
        # add landmarks for goal posts and puck
        goal_posts = [[-0.25, -1.0],
                      [-0.25, 1.0],
                      [0.25, -1.0],
                      [0.25, 1.0]]
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            if i > 0:
                landmark.collide = True
                landmark.movable = False
                landmark.state.p_pos = np.array(goal_posts[i-1])
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.collide = True
                landmark.movable = True
        # add landmarks for rink boundary
        #world.landmarks += self.set_boundaries(world)
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def reset_world(self, world):
        # random properties for landmarks
        self.previous_puck_to_goal_dist = 2.0
        for i, landmark in enumerate(world.landmarks):
            if i > 0:
                landmark.color = np.array([0.7, 0.7, 0.7])
            else:
                landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.index = i
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        world.landmarks[0].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

    # return all agents of the blue team
    def blue_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all agents of the red team
    def red_agents(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on team they belong to
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # reward for blue team agent
        # puck position 
        puck_position = world.landmarks[0].state.p_pos
        agent_position = agent.state.p_pos
        
        # opposition agents 
        red_agents = self.red_agents(world)
        opponent_positions = [red_agent.state.p_pos for red_agent in red_agents]
        
        # Rink boundaries
        rink_bounds = [-2.0, 2.0]
        
        # 1. reward for being near the puck  
        agent_puck_dist = np.linalg.norm(agent_position - puck_position)
        reward = max(0, 1.0 - agent_puck_dist)
        
        # 2. Puck possesion
        possession_reward = 0
        possession_penalty = 0
        for opponent_position in opponent_positions:
            opponent_puck_dist = np.linalg.norm(opponent_position - puck_position)
            if agent_puck_dist <= opponent_puck_dist:
                possession_reward += 0.3
            else:
                possession_penalty -= 0.2
        
        reward += possession_reward + possession_penalty
        
        # 3. Penalty for leaving the rink
        if not (rink_bounds[0] <= agent_position[0] <= rink_bounds[1] and 
            rink_bounds[0] <= agent_position[1] <= rink_bounds[1]):
            reward -= 1.0  
        
        # 4. Reward for scoring the goal
        target_goal = np.array([0.0, -1.0]) # taregt goal for blue team 
        if (target_goal[0] - 0.25 <= puck_position[0] <= target_goal[0] + 0.25 and puck_position[1] <= target_goal[1]):
            reward += 10
        
        # 5. penalty for conceding the goal 
        opponent_target_goal = np.array([0.0, 1.0])
        if (opponent_target_goal[0] - 0.25 <= puck_position[0] <= opponent_target_goal[0] + 0.25 and puck_position[1] >= opponent_target_goal[1]):
            reward -= 5
            
        # 6. reward for reducing distance to the goal       
        target_goal = np.array([0.0, -1.0])  # for blue agent scoring in lower goal
        puck_goal_dist = np.linalg.norm(puck_position - target_goal)
        reward += max(0, self.previous_puck_to_goal_dist - puck_goal_dist)
        self.previous_puck_to_goal_dist = puck_goal_dist
        
        return reward
        
        

    def adversary_reward(self, agent, world):
        # reward for red team agent
        # puck position 
        puck_position = world.landmarks[0].state.p_pos
        agent_position = agent.state.p_pos
        
        # opposition agents (blue agents)
        blue_agents = self.blue_agents(world)
        opponent_positions = [blue_agent.state.p_pos for blue_agent in blue_agents]
        
        # Rink boundaries
        rink_bounds = [-2.0, 2.0]
        
        # 1. reward for being near the puck  
        agent_puck_dist = np.linalg.norm(agent_position - puck_position)
        reward = max(0, 1.0 - agent_puck_dist)
        
        # 2. Puck possession
        possession_reward = 0
        possession_penalty = 0
        for opponent_position in opponent_positions:
            opponent_puck_dist = np.linalg.norm(opponent_position - puck_position)
            if agent_puck_dist <= opponent_puck_dist:
                possession_reward += 0.3  # Reward for possession if closer than opponent
            else:
                possession_penalty -= 0.2  # Penalty for not having possession
        
        reward += possession_reward + possession_penalty
        
        # 3. Penalty for leaving the rink
        if not (rink_bounds[0] <= agent_position[0] <= rink_bounds[1] and 
                rink_bounds[0] <= agent_position[1] <= rink_bounds[1]):
            reward -= 1.0  # Penalty for going out of bounds
        
        # 4. Reward for scoring the goal
        target_goal = np.array([0.0, 1.0]) # taregt goal for red team 
        if (target_goal[0] - 0.25 <= puck_position[0] <= target_goal[0] + 0.25 and puck_position[1] >= target_goal[1]):
            reward += 10
        
        # 5. penalty for conceding the goal 
        opponent_target_goal = np.array([0.0, -1.0])
        if (opponent_target_goal[0] - 0.25 <= puck_position[0] <= opponent_target_goal[0] + 0.25 and puck_position[1] <= opponent_target_goal[1]):
            reward -= 5
            
        # 6. reward for reducing distance to the goal
        target_goal = np.array([0.0, 1.0])  # for red agent scoring in upper goal
        puck_goal_dist = np.linalg.norm(puck_position - target_goal)
        reward += max(0, self.previous_puck_to_goal_dist - puck_goal_dist)
        self.previous_puck_to_goal_dist = puck_goal_dist
        
        return reward
    
    
    def observation(self, agent, world):
        # get positions/vel of all entities in this agent's reference frame
        entity_pos = []
        entity_vel = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            if entity.movable:
                entity_vel.append(entity.state.p_vel)
        # get positions/vel of all other agents in this agent's reference frame
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + entity_pos + entity_vel + other_pos + other_vel)
