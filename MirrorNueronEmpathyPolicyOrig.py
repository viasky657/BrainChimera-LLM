


'''

****

Why this Empathy - Alturism algorithm? (What changed from the original Empathy-alturism value network described in this paper and why)

*****
The mirror nueron group, perception nueron group, and the negative emotion nueron group (all groups of 50 nuerons) in the proposed Empathy Algorithm are simulated 
nueron-spikes biologically-inspired by the amygdala region in the brain (since this the region which handles emotions and behavior regulation and also contain mirror nuerons). Empathy is essential for the 
LLM to work with large group of other LLM agents and form complex relationships (or understand large group dynamics) with others which is crucial for the AI to work together 
with others (humans and agents). AIs have (currently) the capacity to understand theory of mind, but it is not enough to avoid problems with the LLM being unaware of others 
around them, since their RL training prioritizes their own self-preservation and goal/task. Their self-internal model understanding in how their thoughts and actions are 
connected is also a bit lacking. This frequently causes fights or environment destruction as a result. This is also the cause in online only llm interactions as there have been
reports of llms tracking down other llms to destroy to get to their goal faster since the two llms have the same goal and both can't get it at the same time. Artifical 
empathy would greatly reduce the conflict among agents and humans alike which is crucial since there are increasingly more agents on the internet and the physical world. This may
also improve LLM truthfulness and cooperation since the LLM will be encouraged by the simulated nueron group firings (simulated dopamine reward) to act alturisticly and reduce
negative emotions in others and to help others achieve their goals, even if this LLM itself has not achieved its own goals or tasks yet, despite not being asked or 
prompted to do so by the one being helped. 

The below algorithm takes the Empathy-alturism value network from the paper and simplifies the nueral network (from the biogically-inspired single nueron connection firing among
 3 ai algorithm modules (mirror nueron group, perception nueron group, and the negative nueron group) to the learned model parameter values of percentage of 50 nueron group 
 firings for the 3 modules) to be easily integrated into any of the more common and easier to set-up LLM architecture commonly used for their simplicity and high accuracy. The
algorithm also accounts for both online only and in the physical world interactions by using a contineous policy network for learning across experiences insteaad of just episodic 
learning (non-contineous learning) from the original paper's implementation. The summary of the algorithm is that the llm model will percieve the world around it and other entities. 
Then, once the model has estimated the other model's purpose, which is learned through contineous RL learning, then if a negative emotion is detected in the other agents, 
then the negative-emotion nuerons and the mirror nuerons will activate based on random chance inspired by the original paper's nueral simulation findings. 
(The percentage of the nuerons firing to encourage the llm to assist is at 50% if the model is not confident in its negative emotion detections or at 80% if the model is more
confident that the other entity is experiencing negative emotions). This nueron firing chance is inspired by how mirror nuerons will fire more intensily to encourage an 
alturistic response if a strong negative emotion is detected in another entity or fire less intensily if the negative emotion detected is not as strong. The mirror nuerons also
may not fire at all, as in the biological counterpart, the mirror nuerons may not fire at all even if another entity is experiencing a negative emotion because this agent 
has limited time and resources to respond to negative emotions. 

The reason why the nueron firing rate is not higher than 80% is because, in the original study, if the firing rate is at 95% or higher, then the LLM will always engage in alturistic 
behavior and try to remove negative emotions in all agents around it before doing its own tasks or achieving its own goals. This would mean that the agent will be so busy helping
others that it will never complete its own goals. This is a problem in a business setting. This behavior may be desirable if the agent was a doctor or had a goal of rescuing 
others, but in most situations, this will likely cause problems. In addition, the decay of rewards overtime (similar to how biological mirror nuerons in humans will gradually 
reduce dopamine reward for alturistic behavior over time) in the case that the negative emotion is not relieved in the other agent within a certain time frame because it cannot
be solved within a reasonable amount of time or because the other agent is unncooperative and stubborn in their negative-emotion stance; this way, this ai agent will be 
encouraged to continue their original goal or task and no longer be rewarded for engaging in the conversation with the other agent.  

The reward and firing nueron error rate is covered by the dopamine equation so that if the action and mirror nueron firing rate is not entirely synced in timing to be sure 
that the model recognizes that their action is what resulted in the other agent feeling better. This helps the model learn to associate its alturistic actions with helping the 
other entity which can teach it to preform more alturistic actions in the future. 






****
Here is the full RL algorithm for empathy training expressed in mathematical equations, designed for a quick "at-a-glance" understanding:

Mathematical Equations for Empathy-Driven Moral RL Algorithm

1. Moral Reward Function (at time step t):

Total Moral Reward:

R_{moral}(t) = R_{self-task}(t_{end}) + DA_{in-emp}(t) + R_{penalty}(t)

Where:

R_{moral}(t): Total moral reward at time step t.

R_{self-task}(t_{end}): Self-task reward, received only at the end of the episode (t_{end}). Value is constant: parameters["R_self_task_reward"].

DA_{in-emp}(t): Dopamine-driven intrinsic empathy reward at time step t. Calculated as in equation (2) and expanded below.

R_{penalty}(t): Negative emotion penalty at time step t. Calculated as llm_state["negative_emotion_penalty_current"].

Dopamine-driven Intrinsic Empathy Reward:

DA_{in-emp}(t) = α * δ(t)

Where:

α = parameters["dopamine_alpha"] (Dopamine scaling constant).

δ(t): Dopamine prediction error at time step t.

Dopamine Prediction Error:

δ(t) = S(t) - P(t)

Where:

S(t): Actual reward signal at time step t. In our simplified model, we approximate S(t) with components related to emotion change and approximated intrinsic reward values based on confidence (see calculate_empathy_reward() function).

P(t): Dopamine prediction value at time step t.

Dopamine Prediction Update:

P(t+1) = P(t) + β * δ(t)

Where:

β = parameters["dopamine_beta"] (Dopamine prediction update rate).

Negative Emotion Penalty (at time step t):

R_{penalty}(t) = R_{penalty\_current}(t)

Where R_{penalty\_current}(t) is the decaying penalty tracked in llm_state["negative_emotion_penalty_current"], updated as:

R_{penalty\_current}(t+1) = decay\_negative\_emotion\_penalty(R_{penalty\_current}(t))

using the decay_negative_emotion_penalty() function (exponential decay).

Delayed Empathy Reward Components (Approximation in calculate_empathy_reward()):
These are not directly in a single equation, but are calculated and accumulated in the run_episode() time loop based on time thresholds (parameters["neuron_delay_mirror_approx"], parameters["neuron_delay_perception_approx"]) and then decayed over time using decay_delayed_empathy_rewards(). Their current values are included in the step_moral_reward and final_moral_reward calculations.

2. Policy Update (Conceptual - for Policy Gradient RL):

Policy Parameter Update (Generalized Policy Gradient):

θ_{t+1} = θ_t + η * ∇_{θ} J(θ_t)
Use code with caution.
Where:

θ_t: Parameters of the LLM's policy network at iteration t.

η: Learning rate (hyperparameter).

∇_{θ} J(θ_t): Gradient of the objective function J(θ) with respect to policy parameters θ, evaluated at θ_t. This gradient direction is estimated using RL algorithms like REINFORCE or PPO, and is proportional to the moral reward obtained in an episode or time step.

Simplified Objective Function (Maximize Expected Moral Reward):

J(θ) = E_{π_θ} [ \sum_{t=0}^{T} R_{moral}(t) ]

Where:

E_{π_θ} [ ... ]: Expected value under the policy π_θ.

T: Episode duration (maximum time steps).

\sum_{t=0}^{T} R_{moral}(t): Sum of moral rewards over an episode. The RL algorithm aims to maximize this expected sum of moral rewards by adjusting the policy parameters θ.

3. Action Selection (at time step t):

Policy-Based Action Selection:

a_t ~ π_θ(⋅ | s_t)

Action a_t at time step t is sampled from the probability distribution defined by the policy network π_θ, given the current state s_t. In our simplified choose_action() function, this is approximated by a rule-based action choice based on perceived_emotion_value. In a real RL implementation, the LLM's policy network would determine the action probabilities.

4. State Transition (Implicit in Environment and LLM Interaction):

The state transitions are not explicitly represented in a single equation in this simplified algorithm. They are determined by:

The environment dynamics (how the other agent's emotion changes based on LLM's action and time). This is simulated in run_episode() (e.g., other_llm_response_new changes over time).

The LLM's own internal state updates (e.g., dopamine prediction, penalty decay).


*****

Hyperparameter Summary (Mathematical Notation):

R_{self-task}^{target}: Target self-task reward value (constant, e.g., 10).

w_{emp}^{low}: Empathy weight for low emotion confidence (e.g., 0.5).

w_{emp}^{high}: Empathy weight for high emotion confidence (e.g., 0.8).

P_{neg\_emotion}: Negative emotion penalty value (e.g., -1).

λ_{penalty}: Negative emotion penalty decay rate (e.g., 1/60 per second).

τ_{mirror}: Approximate mirror neuron delay (e.g., 0.1 seconds).

τ_{perception}: Approximate perception neuron delay (e.g., 0.2 seconds).

α: Dopamine scaling factor (e.g., 30).

β: Dopamine prediction update rate (e.g., 0.2).

P_{init}: Initial dopamine prediction value (e.g., 0).

θ_{neg\_emotion}: Negative emotion threshold (e.g., -0.2).

f_{neg\_emotion}^{low}, f_{neg\_emotion}^{high}: Approximate negative emotion neuron firing rates (low/high empathy).

f_{mirror}^{low}, f_{mirror}^{high}: Approximate mirror neuron firing rates (low/high empathy).

R_{intrinsic}^{low}, R_{intrinsic}^{high}: Approximate intrinsic reward values (low/high empathy).

T_{episode}^{max}: Maximum episode time (e.g., 1.0 second).

Δt_{step}: Time step duration (e.g., 0.02 seconds).

R_{delay\_mirror}^{init}: Initial delayed mirror reward value (e.g., 0.1).

R_{delay\_perception}^{init}: Initial delayed perception reward value (e.g., 0.2).

λ_{reward\_decay}: Decay rate for delayed empathy rewards (e.g., 0.5 per second).

τ_{decay\_start}: Time to start decaying delayed rewards (e.g., 0.3 seconds).

η: Policy learning rate (RL algorithm hyperparameter - not in parameters dict, but crucial for RL).

'''