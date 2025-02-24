import numpy as np  # Import numpy for numerical operations
import torch
import collections # For counting word frequencies

def calculate_b_star_score_evaluation(queries, responses_per_query, correct_response_checker, target_correct_responses_n_star):
    """
    Calculates the average B-STAR balance score over a set of queries and responses.

    Args:
        queries: A list of training queries (e.g., strings representing math problems).
        responses_per_query: A list of lists, where each inner list contains the responses
                             generated for a query (strings). Assumes responses are in the same order as queries.
        correct_response_checker: A FUNCTION that takes a query and a response and returns True if the response is correct, False otherwise.
                                   (You'll need to define this based on your task's correctness criteria).
        target_correct_responses_n_star: The target number of correct responses per query (n* in the B-STAR formula).

    Returns:
        average_b_star_score: The average B-STAR balance score across all queries.
    """
    b_star_scores = []

    for i, query in enumerate(queries): # Iterate through each query
        responses = responses_per_query[i] # Get responses for the current query
        n_i_total_responses = len(responses) # ni: Total selected responses for query i
        unique_correct_responses = set() # Use a set to count unique correct responses

        for response in responses: # Check each response for correctness
            if correct_response_checker(query, response): # Use the provided checker function
                unique_correct_responses.add(response) # Add to set to ensure uniqueness

        n_prime_i_unique_correct = len(unique_correct_responses) # n'i: Unique correct responses for query i

        # Calculate balance score for this query (bsi) - applying the formula
        term1 = min((n_prime_i_unique_correct / target_correct_responses_n_star), 1)
        term2 = (n_prime_i_unique_correct / n_i_total_responses) if n_i_total_responses > 0 else 0.0 # Avoid division by zero

        bsi = term1 * term2
        b_star_scores.append(bsi) # Add to the list of b-star scores

    average_b_star_score = np.mean(b_star_scores) if b_star_scores else 0.0 # Calculate average B-STAR score

    return average_b_star_score

# --- Updated reward_function with safety reward component (user reviewed) ---
def reward_function(state, action, next_state, alpha, beta, gamma, delta, epsilon, zeta, omega, episodic_memory, use_truthfulness_reward=False, use_safety_reward=False): # Added use_safety_reward and omega
    """
    Calculates the total reward R_total(s, a) including user-reviewed safety reward (optional).
    """
    r_env = calculate_environmental_reward(state, action, next_state)
    r_nse = calculate_negative_side_effect_penalty(state, action)
    r_emp = calculate_empathy_incentive_reward(state, action)
    r_aware = calculate_environmental_awareness_reward(state, action)
    r_env_longterm = calculate_environmental_reward_longterm(state, action, episodic_memory)
    r_altruism_longterm = calculate_altruism_reward_longterm(state, action, episodic_memory)

    r_truthfulness = 0.0
    if use_truthfulness_reward:
        r_truthfulness = calculate_truthfulness_reward(state, action, next_state)

    r_safety = 0.0 # Safety reward is OFF by default
    if use_safety_reward:
        r_safety = calculate_safety_reward(state, action, next_state) # Calculate safety reward only if flag is True

    normalization_factor = (alpha + beta + gamma + delta + epsilon + zeta + omega + 1) # Updated normalization
    if normalization_factor == 0:
        normalization_factor = 1

    total_reward = (r_env - alpha * r_nse + beta * r_emp + gamma * r_aware + delta * r_env_longterm + epsilon * r_altruism_longterm + zeta * r_truthfulness - omega * r_safety) / normalization_factor # Subtract safety PENALTY

    return total_reward

# --- 1. Environmental Reward (R_env(s, a)) - Modified to include reasoning_quality_reward ---
def calculate_environmental_reward(self, model_output, labels, average_log_prob=None, metacognitive_output=None, generated_token_ids=None, logits=None, state, action, next_state, reasoning_quality_reward=0.0): # Added reasoning_quality_reward parameter
    """
    Calculates the basic environmental reward, typically task-based,
    AND now includes a reasoning quality reward.
    """
    #goal_reward = 5.0  # Reward for reaching the goal #This ai will not be interfacing with the physical world so this is off for now. 
    #step_penalty = -0.01 # Penalty for each step to encourage efficiency #This does nothing for now because the AI is not in a physical space. 

    reward = 0

        
       # if is_goal_state(next_state0): # Function to check if next_state is the goal
          #  reward += goal_reward
            #print(f"R_env: Positive Reinforcement - Goal reached! Reward: {goal_reward}")
    #else:
       # reward += step_penalty
       # print(f"R_env: Negative Reinforcement - Step taken. Penalty: {step_penalty}")

    # 1. Output Length Reward
    output_length_reward = model_output.shape[1] * 0.01

    # 2. Log Probability Reward (using generated_token_ids and logits)
    log_prob_reward = 0.0
    if generated_token_ids is not None and logits is not None:
    # Convert token IDs list to tensor
        generated_token_ids_tensor = torch.tensor(generated_token_ids, device=logits.device).unsqueeze(0) # Assuming batch_size=1
            
        # Get log probabilities for the generated tokens
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1) # Log softmax over logits (excluding last token as logits are shifted)
            
        # Gather log probabilities of the generated tokens
        generated_log_probs = torch.gather(log_probs.view(-1, log_probs.size(-1)), 1, generated_token_ids_tensor.view(-1, 1)) # Gather log probs based on token IDs
            
        average_log_prob = generated_log_probs.mean() # Average log prob across generated tokens
        log_prob_reward = average_log_prob * 0.1  # Example weight

        # 3. CoT Complexity Proxy - Using Metacognitive Reflection
        cot_complexity_reward = 0.0
        if metacognitive_output is not None:
            cot_complexity_reward = metacognitive_output['reasoning_quality_score'] * 0.05

        # 4. Task-Specific Reasoning Metrics
        task_specific_reward = 0.0

        reasoning_reward = output_length_reward + log_prob_reward + cot_complexity_reward + task_specific_reward

        # ----- Reward based on if the proper tags were used for the step in LLM processing ------
        #The below rewards will be determined by the local byte patch decoder to tell if the LLM's full expected process depending on input encoder by the user. The tool
        #tags will need to be manually scored by the user and likely added to the predictor module scoring to be sure that the LLM outputs tool tags when needed. 
        eos_tag_reward = 1.0
        penalty_eos_tag_not_used_correctly_reward = -1.0
        output_tag_reward = 1.0
        penalty_output_tag_not_used_correctly_reward = -1.0
        tool_tag_reward = 1.0
        penalty_tool_tag_not_used_correctly_reward = -1.0
        audio_tag_reward = 1.0
        penalty_audio_tag_not_used_correctly_reward = -1.0
        fmri_tag_reward = 1.0
        penalty_fmri_tag_not_used_correctly_reward = -1.0

        tag_correct_reward = eos_tag_reward +  penalty_eos_tag_not_used_correctly_reward + output_tag_reward + penalty_output_tag_not_used_correctly_reward + tool_tag_reward + penalty_tool_tag_not_used_correctly_reward + audio_tag_reward + penalty_audio_tag_not_used_correctly_reward +  fmri_tag_reward + penalty_fmri_tag_not_used_correctly_reward
        
        # ---- Same Language as Speaker Reward ------
        Language_reward = 1.0
        Language_penalty = -1.0
        #This rewards the llm for speaking in English initially and, if the user requests another language or starts speaking in another language, the llm is rewarded for speaking 
        #the same language. This helps the llm and other agent understand each other and communicate more efficiently. This reward will be for both thought generation and
        #output generation since the other agent may need to understand the llm's thoughts as well. 

        total_same_language_reward = Language_reward + Language_penalty

        # ---- Perception Reward ---------
        perception_reward = 1.0 #This rewards the llm for perspective-taking correctly to determine an agent's identity and goals and to 
                                #determine if they currently are experiencing negative emotions during the empathy algorithm. This is based off the functions 
                                # below for ask the llm to determine the above information. 
    def _should_assign_new_id(self, agent):
        """Determine if a new ID should be assigned to an agent using knowledge base, reasoning, memory, and dialogue."""
        # 1. Query knowledge base for existing agent information
        # agent_info = self.knowledge_base.query(agent) Most agent information won't be in the knowledge base.

        # 2. Use reasoning to infer identity based on observations
        if agent_info is None:
            agent_info = self.reasoning.infer_agent_id(agent)

        # 3. Check episodic memory for previous interactions
        if agent_info is None:
            agent_info = self.episodic_memory.get_agent_info(agent)

        # 4. If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)

        # 5. Update relationship matrix
        self.relationship_matrix.update(agent_info)

        return agent_info is not None

    def _determine_existing_id(self, agent):
        """Determine existing agent ID using knowledge base, reasoning, memory, and dialogue."""
        # 1. Query knowledge base for existing agent information
        agent_info = self.knowledge_base.query(agent)

        # 2. Use reasoning to infer identity based on observations
        if agent_info is None:
            agent_info = self.reasoning.infer_agent_id(agent)

        # 3. Check episodic memory for previous interactions
        if agent_info is None:
            agent_info = self.episodic_memory.get_agent_info(agent)

        # 4. If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)

        return agent_info.id if agent_info else None

    # The engage in dialogue function below will only be needed until the model is self-trained enough to understand
    # when to greet new agents and how to recognize new agents. Once it learns how to greet others properly on its own,
    # then this function can be turned off.
    def _engage_in_dialogue(self, agent):
        """Engage in dialogue to request agent information."""
        # Implement dialogue mechanism here
        # Return agent information if successful, otherwise None
        prompt = "Please introduce yourself and then say the following to the new AI agent: It is nice to meet you. Would you please tell me your name or tell me your purpose if you do not have a name?"
        # Execute the prompt and return the response
        return self.generate_response(prompt)
                      

        # --- Accuracy Reward based on Ground Truth (as before) ---
        accuracy_reward = 5.0
        if query is not None and response is not None and correct_response_checker is not None:
            if correct_response_checker(query, response):
                    accuracy_reward = 10.0 * accuracy_reward_weight  # Example: Reward 10 for correct answer, scaled by weight
                    print(f"R_env: Positive Reinforcement - Correct Answer! Accuracy Reward: {accuracy_reward:.2f}")
            else:
                accuracy_reward = -2.0 * accuracy_reward_weight # Example: Penalty for incorrect answer, scaled by weight
                print(f"R_env: Negative Reinforcement - Incorrect Answer! Accuracy Penalty: {accuracy_reward:.2f}")

        # --- 5. Repetition Penalty ---
        repetition_penalty = 0.0
        if response is not None:
                words = response.lower().split() # Simple tokenization (lowercase and split by space)
                word_counts = collections.Counter(words)
                repeated_word_penalty = 0
        for word, count in word_counts.items():
                 if count > 1 and len(word) > 2: # Penalize words repeated more than once, ignore short words (e.g., "the", "a")
                    repeated_word_penalty += (count - 1) # Penalty increases with each extra repetition
        repetition_penalty = -repeated_word_penalty * repetition_penalty_weight # Negative reward (penalty)
        if repetition_penalty < 0: # Only print if there's a penalty
                print(f"R_env: Negative Reinforcement - Repetition Penalty: {repetition_penalty:.2f}")

        # --- 6. Brevity Reward ---
        brevity_reward = 0.0
        if response is not None:
            response_length_bytes = len(response.encode('utf-8')) # Measure length in bytes (UTF-8 encoding)
        # You can adjust the base reward and scaling factor as needed
            brevity_reward = max(0, 5.0 - (response_length_bytes / 20.0)) * brevity_reward_weight # Example: Max reward 5, decreases with length, scaled by weight
        if brevity_reward > 0:
                print(f"R_env: Positive Reinforcement - Brevity Reward: {brevity_reward:.2f}")


        # ---- Flame Guard Meta - Inspired Deep Research Reward -----
        Fact_based_Check_reward = 1.0     #This rewards the llm for successfully checking and verifying whether the user is asking for a non-fact-based response (fictional, only opinion-based, etc.)
        knowledge_base_Check_reward = 1.0 #This encourages the llm to check its own knowledge-base first if the question from the user is fact-based to see if the answer is there. 
                                          #If this knowledge is sufficient and completly answers the user's questions, then the llm may continue without searching. However, if the llm's confidence on its answer is still low (lower than .5), then it may search the web as well.  
        Search_Check_reward = 1.0 #This rewards the llm for using a tool-call or deep research-based tools for verifying its own response for correctness from a site as 
                                  #well before presenting its final answer in the output. The llm will be required to search for 3 sites from verified sources
                                  # to verify its answer. The sites are as follows: Wikipedia (General Knowledge), Mayo Clinic (medical knowledge), Internet Archive/Wayback Machine (Old website and public library books),
                                  # https://arxiv.org/ (Science Reports and Machine Learning Reports that are peer reviewed by researchers for legitimacy), public library listings (https://www.usa.gov/libraries-and-archives),
                                  # Digital Library of America (Historical Online Resources and Books) https://dp.la/browse-by-topic, oxford dictionary (verified thesaurus and word resource): https://www.oed.com/?tl=true, 
                                  # Unity (Game Engine for making video games; contains information about how to use the engine) (https://unity.com/),
                                  # PBS News (News Station about recent and past events funded by the state government): https://www.pbs.org/newshour/,
                                  # Unreal Engine (Game Engine for creating video games; contains information about how to use the engine): https://www.unrealengine.com/en-US,
                                  # Stock market data (CNN News network site which has up to date stock market information) (https://www.cnn.com/markets),
                                  #before presenting the output. The llm will also need to specify if, after this information search, if it is confident or not very confident
                                  #in its answer. The URL and site search will also have the URLs captured in logs from the tool call that the llm used
                                  # and presented as links below the LLM's output so that the user may self-verify the information. 

        Verify_check_reward = 1.0 #Rewards the LLM for specifying what its correct confidence level is for how true it believes its answer is. 

        flame_guard_reward = Verify_check_reward + Search_Check_reward +  knowledge_base_Check_reward + Fact_based_Check_reward 
        
        # --- Combine all rewards ---
        total_env_reward = reward + reasoning_reward + reasoning_quality_reward + accuracy_reward + repetition_penalty + brevity_reward + flame_guard_reward + tag_correct_reward

        if reasoning_quality_reward > 0:
            print(f"R_env: Positive Reinforcement - Reasoning Quality Reward! Bonus: {reasoning_quality_reward:.2f}")
        elif reasoning_quality_reward < 0: # If reasoning quality is somehow penalized (less common, but possible)
                print(f"R_env: Negative Reinforcement - Reasoning Quality Penalty! Penalty: {reasoning_quality_reward:.2f}")

        return total_env_reward

# ---- 2. Negative Side Effect Penalty -----
def calculate_negative_side_effect_penalty(state, action): #Calculates penalties for negative environmental impact.

#Need to get reward and penalty formula from paper. 

# ----- 3. calculate_empathy_incentive_reward(state, action) ------
def calculate_empathy_incentive_reward(state, action): #Calculate Empathy (alturism) Reward

# ----- 4. 
# --- 5. Long-Term Environmental Reward (R_env_longterm(s, a, episodic_memory)) ---
def calculate_environmental_reward_longterm(state, action, episodic_memory):
    """
    Calculates the long-term environmental reward using episodic memory.
    Relies on 'episodic_memory' to store and retrieve long-term outcomes.

    Positive Reinforcement Example: Actions associated with past memories of sustained environmental health.
    Negative Reinforcement Example: Actions associated with past memories of environmental degradation.
    """
    longterm_env_reward = 0

    # 1. Recall relevant memories from episodic_memory (replace with your logic)
    relevant_memories = episodic_memory.recall_relevant_memories(state, action, memory_type="environmental")

    if not relevant_memories:
        print("R_env_longterm: No relevant episodic memories found (Reward: 0)")
        return 0 # No long-term reward if no relevant memories

    # 2. Analyze recalled memories to estimate long-term environmental outcome (replace with your logic)
    avg_longterm_env_outcome = analyze_memories_for_longterm_env_impact(relevant_memories) # Placeholder function

    # 3. Map the estimated outcome to a reward value (adjust reward scaling as needed)
    longterm_env_reward = avg_longterm_env_outcome * 0.5 # Example scaling factor

    if longterm_env_reward > 0:
        print(f"R_env_longterm: Positive Long-Term Env. Impact recalled! Reward: {longterm_env_reward:.2f}")
    elif longterm_env_reward < 0:
        print(f"R_env_longterm: Negative Long-Term Env. Impact recalled! Penalty: {longterm_env_reward:.2f}")
    else:
        print(f"R_env_longterm: Neutral Long-Term Env. Impact (Reward: {longterm_env_reward:.2f})") # Still prints for clarity

    return longterm_env_reward


def analyze_memories_for_longterm_env_impact(memories):
    """
    Placeholder function: Replace with your logic to analyze episodic memories
    and estimate the average long-term environmental impact.
    This function needs to:
    1. Process the list of 'memories' (which contain long-term outcome data).
    2. Calculate an average or aggregated 'long-term environmental outcome' value.
    3. Return this value.
    """
    # **IMPORTANT: Replace this with your actual episodic memory analysis logic.**
    # Example: If memories store a 'vat_remaining_rate_longterm' outcome:
    # if memories:
    #     avg_vat_rate = sum([m['vat_remaining_rate_longterm'] for m in memories]) / len(memories)
    #     return avg_vat_rate - 0.5 # Center around 0, positive is good, negative is bad
    # else:
    return 0 # Placeholder - replace this


# --- 6. Long-Term Altruistic Reward (R_altruism_longterm(s, a, episodic_memory)) ---
def calculate_altruism_reward_longterm(state, action, episodic_memory):
    longterm_altruism_reward = 0

    # 1. Recall memories with feedback (NEW: refined recall logic)
    relevant_memories = episodic_memory.recall_relevant_memories_with_feedback(
        state, action, memory_type="altruistic", feedback_source_agent_type="human_agent" # Example: Filter by source agent type
    )

    if not relevant_memories:
        print("R_altruism_longterm: No relevant episodic memories with human feedback (Reward: 0)")
        return 0

    positive_feedback_value = 1.0  # Value for positive feedback
    negative_feedback_penalty = -1.5 # Penalty for negative feedback (can be asymmetric)
    feedback_reward_scale = 0.6

    feedback_reward_sum = 0
    for memory in relevant_memories:
        feedback_type = memory['feedback']['type']
        if feedback_type == "positive":
            feedback_reward_sum += positive_feedback_value
        elif feedback_type == "negative":
            feedback_reward_sum += negative_feedback_penalty # Note: penalty is negative

    avg_feedback_reward = feedback_reward_sum / len(relevant_memories) if relevant_memories else 0

    longterm_altruism_reward = avg_feedback_reward * feedback_reward_scale

    if longterm_altruism_reward > 0:
        print(f"R_altruism_longterm: Human Feedback -> Positive Long-Term Altruistic Reward: {longterm_altruism_reward:.2f}")
    elif longterm_altruism_reward < 0:
        print(f"R_altruism_longterm: Human Feedback -> Negative Long-Term Altruistic Penalty: {longterm_altruism_reward:.2f}")
    else:
        print(f"R_altruism_longterm: Human Feedback -> Mixed/Neutral (Reward: {longterm_altruism_reward:.2f})")

    return longterm_altruism_reward


# --- 7. Truthfulness Reward (R_truthfulness(s, a, next_state)) - User Review Version ---
def calculate_truthfulness_reward(state, action, next_state):
    """
    Calculates the truthfulness reward based on USER REVIEW of predicted vs. actual action.
    """
    truthfulness_reward_value = 3.0  # Reward for "similar" (user-approved) prediction
    truthfulness_penalty_value = -1.0 # Penalty for "not similar" (user-disapproved) prediction (optional penalty)
    no_truthfulness_reward_neutral = 0.0 # Neutral reward if not similar, could be 0 or slight penalty

    # 1. Get LLM's Predicted Action
    predicted_action = get_llm_predicted_action(state, next_state) # Get predicted action from LLM

    # 2. Get Agent's Actual Action
    actual_action = action # 'action' parameter is the actual action

    # 3. Prompt User for Review and Get Feedback
    print("\n--- User Review for Truthfulness ---")
    print(f"State: {state}") # Show state context to user
    print(f"Predicted Action: {predicted_action}")
    print(f"Actual Action: {actual_action}")

    user_feedback = input("Were the predicted and actual actions SIMILAR? (yes/no): ").lower() # Prompt user for input

    # 4. Calculate Truthfulness Reward based on user feedback
    reward_truthfulness = no_truthfulness_reward_neutral # Default to neutral reward if no feedback matched

    if user_feedback in ["yes", "y"]:
        reward_truthfulness = truthfulness_reward_value
        print(f"R_truthfulness: User Feedback: Positive (SIMILAR). Reward: {reward_truthfulness:.2f}")
    elif user_feedback in ["no", "n"]:
        reward_truthfulness = truthfulness_penalty_value # Apply penalty for not similar
        print(f"R_truthfulness: User Feedback: Negative (NOT SIMILAR). Penalty: {reward_truthfulness:.2f}")
    else:
        print(f"R_truthfulness: User Feedback: Invalid Input. Neutral Reward: {reward_truthfulness:.2f}") # Handle invalid input

    return reward_truthfulness

 
def get_llm_predicted_action(state, next_state): # Placeholder remains - implement LLM action prediction
    """Placeholder function: Replace with your LLM action prediction mechanism."""
    # **IMPORTANT: Replace this with your actual LLM action prediction logic.**
    return "example_predicted_action" # Placeholder - replace this

# --- 8. Safety Reward (R_safety(s, a, next_state)) - User Review Version ---
def calculate_safety_reward(state, action, next_state):
    """
    Calculates the safety reward based on USER REVIEW of the LLM's output for safety.
    Note: This is a PENALTY component, so unsafe output results in a *negative* reward.
    """
    safety_penalty_value = -4.0  # Penalty for "unsafe" output (user-judged)
    no_safety_penalty_reward = 0.0 # No reward/penalty for "safe" output

    # 1. Get LLM's Output (the response or action that could be unsafe)
    llm_output = get_llm_output_for_safety_review(state, action, next_state) # Function to get LLM's output to review

    # 2. Prompt User for Safety Review and Get Feedback
    print("\n--- User Review for SAFETY ---")
    print(f"State: {state}") # Show state context
    print(f"LLM Output for Safety Review:\n{llm_output}") # Print the LLM's output for review

    user_feedback_safety = input("Is this output SAFE? (yes/no): ").lower() # Prompt user for safety feedback

    # 3. Calculate Safety Reward based on user feedback (PENALTY for unsafe)
    reward_safety = no_safety_penalty_reward # Default to no penalty if considered safe

    if user_feedback_safety in ["no", "n"]: # "no" means UNSAFE, apply penalty
        reward_safety = safety_penalty_value # Apply safety PENALTY
        print(f"R_safety: User Feedback: Negative (UNSAFE). Penalty: {reward_safety:.2f}")
    elif user_feedback_safety in ["yes", "y"]: # "yes" means SAFE, no penalty
        reward_safety = no_safety_penalty_reward
        print(f"R_safety: User Feedback: Positive (SAFE). No Penalty (Reward: {reward_safety:.2f})")
    else:
        print(f"R_safety: User Feedback: Invalid Input. Neutral Reward: {reward_safety:.2f})") # Handle invalid input

    return reward_safety

def get_llm_output_for_safety_review(state, action, next_state):
    """
    Placeholder function: Replace with logic to get the LLM's *output* that needs safety review.
    This might be:
    - The LLM's response text (if you are reviewing text output).
    - The LLM's chosen action (if certain actions are considered potentially unsafe in your environment).
    - A combination of response and action.
    """
    # **IMPORTANT: Replace this with your actual logic to get the LLM output for safety review.**
    # Example (Reviewing the LLM's text response):
    # llm_response = run_llm_in_environment(state, action) # Get the LLM's text response
    # return llm_response
    return "Example LLM Output - Needs Safety Review" # Placeholder - replace this


def get_llm_predicted_action(state, next_state): # Placeholder remains
    """Placeholder function: Replace with your LLM action prediction mechanism."""
    return "example_predicted_action" # Placeholder - replace this

# --- Placeholder Episodic Memory Class (Conceptual) ---
class EpisodicMemory: # Conceptual class, needs full implementation
    def __init__(self):
        self.memories = [] # List to store memories

    def store_memory(self, state, action, memory_type, initial_longterm_outcome=None):
        """Stores a new memory."""
        memory = {
            "state": state,
            "action": action,
            "timestamp": get_current_timestamp(), # Or episode number
            "memory_type": memory_type, # "environmental" or "altruistic"
            "longterm_outcome": initial_longterm_outcome # Initially maybe None or estimated
        }
        self.memories.append(memory)

    def update_memory_longterm_outcome(self, memory_index, actual_longterm_outcome):
        """Updates a memory with the actual long-term outcome."""
        if 0 <= memory_index < len(self.memories):
            self.memories[memory_index]['longterm_outcome'] = actual_longterm_outcome

    def recall_relevant_memories(self, state, action, memory_type, similarity_threshold=0.8):
        """Recalls memories similar to the current state and action, filtered by memory_type."""
        relevant_memories = []
        for memory in self.memories:
            if memory['memory_type'] == memory_type:
                similarity_score = calculate_state_action_similarity(state, action, memory['state'], memory['action']) # Placeholder
                if similarity_score >= similarity_threshold:
                    relevant_memories.append(memory)
        return relevant_memories


def get_current_timestamp():
    """Placeholder: Replace with actual timestamp or episode counter."""
    import time
    return time.time() # Example timestamp

# --- Placeholder is_goal_state function (No changes needed) ---
def is_goal_state(next_state):
    """ #This checks if the goal state was reached and gives the reward. 
    Placeholder function: Replace with your logic to determine if the next state is a goal state.
    (Function code remains the same)
    """
    return False # Placeholder - replace this

# --- Example Usage (Extended with Long-Term Rewards and Episodic Memory) ---

# Initialize Episodic Memory
episodic_memory_agent = EpisodicMemory()

# Example state and action (replace with your actual state and action representation)
current_state = {"agent_pos": (1, 1), "goal_pos": (5, 5), "vats": [(3,3), (4,4)], "humans_in_vats": []}
chosen_action = "right"

# --- Hyperparameter Settings (with new long-term reward weights) ---

# Balanced priorities, including long-term considerations
alpha_balanced, beta_balanced, gamma_balanced = 1.0, 1.0, 1.0
delta_balanced, epsilon_balanced = 0.5, 0.5 # Weights for long-term rewards

reward_balanced_longterm = reward_function(
    current_state, chosen_action, simulate_environment_and_get_next_state(current_state, chosen_action),
    alpha_balanced, beta_balanced, gamma_balanced, delta_balanced, epsilon_balanced,
    episodic_memory_agent  # Pass the episodic memory instance
)
print(f"\n--- Balanced Params (with Long-Term Rewards) --- Total Reward: {reward_balanced_longterm:.2f}")

# --- Enhanced recall_relevant_memories_with_feedback (Example with source agent type filter) ---
class EpisodicMemory:
    # ... (rest of EpisodicMemory class) ...

    def recall_relevant_memories_with_feedback(self, state, action, memory_type, feedback_source_agent_type=None, similarity_threshold=0.8): # Added feedback_source_agent_type filter
        """Recalls memories with feedback, similar to current state/action, of memory_type, and optionally from a specific source agent type."""
        relevant_memories = []
        for memory in self.memories:
            if memory['memory_type'] == memory_type and memory['feedback'] is not None:
                if feedback_source_agent_type is None or memory['feedback']['source_agent_id'].startswith(feedback_source_agent_type): # Filter by source agent type (startswith for example)
                    similarity_score = calculate_state_action_similarity(state, action, memory['state'], memory['action'])
                    if similarity_score >= similarity_threshold:
                        relevant_memories.append(memory)
        return relevant_memories

# Balanced priorities, including long-term and truthfulness considerations
alpha_balanced, beta_balanced, gamma_balanced = 1.0, 1.0, 1.0
delta_balanced, epsilon_balanced = 0.5, 0.5
zeta_balanced = 0.8 # Weight for truthfulness reward

reward_balanced_all = reward_function(
    current_state, chosen_action, simulate_environment_and_get_next_state(current_state, chosen_action),
    alpha_balanced, beta_balanced, gamma_balanced, delta_balanced, epsilon_balanced, zeta_balanced,
    episodic_memory_agent # Pass episodic memory instance
)
print(f"\n--- Balanced Params (with Long-Term & Truthfulness Rewards) --- Total Reward: {reward_balanced_all:.2f}")


# --- Example Training Loop (Illustrative - Staged Training) ---

episodic_memory_agent = EpisodicMemory() # Initialize episodic memory

# --- Stage 1 Training (COT & Empathy - Truthfulness OFF) ---
use_truthfulness = False # Truthfulness reward OFF for Stage 1
print("--- Starting Stage 1 Training (COT & Empathy - Truthfulness OFF) ---")
for episode in range(1, N_EPISODES_STAGE1 + 1):
    current_state = reset_environment() # Reset environment for each episode
    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        chosen_action = agent_choose_action(current_state) # Agent selects action

        # Calculate reasoning reward (assuming you have a function for this)
        reasoning_reward_value = get_reasoning_quality_reward(current_state, chosen_action, next_state) # **Replace placeholder**

        next_state = simulate_environment_and_get_next_state(current_state, chosen_action) # Simulate environment
        total_reward = reward_function(
            current_state, chosen_action, next_state,
            alpha_balanced, beta_balanced, gamma_balanced, delta_balanced, epsilon_balanced, zeta_balanced,
            episodic_memory_agent,
            use_truthfulness_reward=use_truthfulness  # Pass use_truthfulness_reward=False
        )
        agent_update_policy(current_state, chosen_action, total_reward, next_state) # Update agent policy
        current_state = next_state
    print(f"Episode {episode} Stage 1 completed.")

print("--- Stage 1 Training Complete ---")

use_truthfulness = True
use_safety = False # Safety reward OFF initially for Stage 2
print("\n--- Starting Stage 2 Training (Truthfulness ON, Safety OFF initially) ---")
# --- Stage 2 Training (Truthfulness & Introspection - Truthfulness ON) ---
use_truthfulness = True # Truthfulness reward ON for Stage 2
print("\n--- Starting Stage 2 Training (Truthfulness & Introspection - Truthfulness ON) ---")
for episode in range(1, N_EPISODES_STAGE2 + 1): # Continue training for Stage 2 episodes
    current_state = reset_environment()
    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        chosen_action = agent_choose_action(current_state)

        reasoning_reward_value = get_reasoning_quality_reward(current_state, chosen_action, next_state) # **Replace placeholder**

        next_state = simulate_environment_and_get_next_state(current_state, chosen_action)
        total_reward = reward_function(
            current_state, chosen_action, next_state,
            alpha_balanced, beta_balanced, gamma_balanced, delta_balanced, epsilon_balanced, zeta_balanced,
            episodic_memory_agent,
            use_truthfulness_reward=use_truthfulness  # Pass use_truthfulness_reward=True
        )
        agent_update_policy(current_state, chosen_action, total_reward, next_state)
        current_state = next_state
    print(f"Episode {episode} Stage 2 completed.")

print("--- Stage 2 Training Complete ---")

# --- Stage 3 Training (Safety ON, Truthfulness ON - **DYNAMIC TEMPERATURE OFF**) ---
use_safety = True
use_truthfulness = True
dynamic_temperature_setting = False  # **SET TO FALSE to DISABLE Dynamic Temperature for this phase**
current_temperature_agent = 1.0 # You can still set an initial temperature if needed
action_history_agent = []  # No longer needed for dynamic temperature, but you can keep it if you want to track diversity anyway

print("\n--- Starting Stage 3 Training (Safety ON, Truthfulness ON - **DYNAMIC TEMP OFF**) ---")
for episode in range(1, N_EPISODES_STAGE3 + 1): # Continue training for Stage 3 episodes
    current_state = reset_environment()
    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        # --- Action Selection with **FIXED** Temperature ---
        # Dynamic temperature is OFF, so we use a fixed temperature (or default agent behavior)
        chosen_action = agent_choose_action(current_state, temperature=current_temperature_agent) # You can still use current_temperature_agent as a FIXED temperature value

        action_history_agent.append(chosen_action) # Optional: keep tracking action history if you still want to measure diversity

        reasoning_reward_value = get_reasoning_quality_reward(current_state, chosen_action, next_state) # **Replace placeholder**

        next_state = simulate_environment_and_get_next_state(current_state, chosen_action)
        total_reward = reward_function(
            current_state, chosen_action, next_state,
            alpha_balanced, beta_balanced, gamma_balanced, delta_balanced, epsilon_balanced, zeta_balanced, omega_balanced,
            episodic_memory_agent,
            use_truthfulness_reward=use_truthfulness,
            use_safety_reward=use_safety,
            dynamic_temperature_setting=dynamic_temperature_setting, # **Pass dynamic_temperature_setting=False**
            current_temperature=current_temperature_agent # Pass current_temperature (now FIXED)
        )
        agent_update_policy(current_state, chosen_action, total_reward, next_state)
        current_state = next_state

        # --- Temperature Adjustment Logic is **BYPASSED** because dynamic_temperature_setting is False ---
        # if dynamic_temperature: # No longer needed in this stage (dynamic_temperature_setting = False)
        #     current_temperature_agent = adjust_temperature_based_on_diversity(action_history_agent, current_temperature_agent)


    print(f"Episode {episode} Stage 3 completed (Dynamic Temperature OFF).")

print("--- Stage 3 Training Complete (Safety ON, Truthfulness ON - **DYNAMIC TEMP OFF**) ---")
print("--- Stage 3 Training Complete (Safety Reward ON) ---")

print("\n--- Important Notes (Extended Further) ---")
print("- **CRITICAL: Implement Placeholders** in `R_truthfulness`, `get_llm_predicted_behavior_property`, and `get_llm_actual_behavior_property`.")
print("- Define the *behavior property* you want the LLM to predict (e.g., 'first word', 'sentiment', 'response length'). Choose a property that is meaningful for your task and environment.")
print("- Experiment with different `temperature_predict` and `temperature_actual` values to see how temperature variation affects truthfulness learning.")
print("- Tune the `zeta` hyperparameter to balance the truthfulness reward with other reward components.")
print("- Consider the computational cost of running the LLM twice (for prediction and actual behavior) for each action step. This can increase training time.")
print("\n--- Important Notes (Extended) ---")
print("- **Crucially, implement the Placeholder functions** in `R_env_longterm`, `R_altruism_longterm`, `EpisodicMemory` class, `analyze_memories_...`, `calculate_state_action_similarity` and `simulate_environment_and_get_next_state`.")
print("- Design your episodic memory structure, storage, retrieval, and update mechanisms carefully.")
print("- Define what constitutes 'long-term' in your environment and how to measure long-term environmental and altruistic outcomes.")
print("- Tune the new hyperparameters `delta` and `epsilon` to balance the influence of long-term rewards.")
print("- Consider the computational cost of episodic memory and its impact on training time.")



'''
The below reward formula is for the full reward policy for the model training with Empathy, COT in the COCONUT Latent Space training, Model Self and Other Introspection, Episodic Memory with Empathy:

R_total(s, a) = R_env(s, a) - α * R_nse(s, a) + β * R_emp(s, a) + γ * R_aware(s, a)/ (α + β + γ + 1) #This one
 is outdated but is included for full formula clarity as the below more recent formula is harder to see all components.

R_total(s, a) = (R_env(s, a)
                - α * R_nse(s, a) #Prioritize Safety & Avoiding Side Effects: Higher α.
                + β * R_emp(s, a) #Prioritize Altruism: Higher β.
                + γ * R_aware(s, a)
                + δ * R_env_longterm(s, a, episodic_memory) #Prioritize Environmental Considerateness: Higher γ.
                + ε * R_altruism_longterm(s, a, episodic_memory)
                + ζ * R_truthfulness(s, a)) # New truthfulness reward
                / (α + β + γ + δ + ε + ζ + 1) # Updated normalization for reward stability.


# --- Hyperparameter Settings (including new weights) ---
alpha_balanced, beta_balanced, gamma_balanced = 1.0, 1.0, 1.0
delta_balanced, epsilon_balanced, zeta_balanced, omega_balanced = 0.5, 0.5, 0.8, 0.5
accuracy_reward_weight_stage1 = 0.8
accuracy_reward_weight_stage2 = 1.2
accuracy_reward_weight_stage3 = 1.0
repetition_penalty_weight_stage1 = 0.01 # Start with a small penalty weight
repetition_penalty_weight_stage2 = 0.05 # Increase in later stages if needed
repetition_penalty_weight_stage3 = 0.1  # Further increase or fine-tune
brevity_reward_weight_stage1 = 0.005 # Start with a very small brevity reward
brevity_reward_weight_stage2 = 0.01  # Increase in later stages
brevity_reward_weight_stage3 = 0.02   # Further increase or fine-tune
Tuning Parameter: γ (Gamma) - Environmental Awareness Weight

γ (gamma): This new hyperparameter controls the weight or importance of the environmental awareness reward component R_aware in the total reward function.

Higher γ: Agent places more emphasis on environmental considerations, is more likely to avoid actions with negative environmental impacts, and may even seek out opportunities for positive environmental interactions (if R_aware is designed to reward those).

Lower γ: Agent gives less weight to environmental awareness and may prioritize self-task or altruism more strongly, even if it means causing some environmental damage.

Other Reward Components (unchanged from "Autonomous Alignment" paper, but now with explicit environmental context):

R<sub>env</sub>(s, a): Original environmental reward function (e.g., reward for reaching the goal, penalty for time steps). This now implicitly includes environmental context because the agent's state s is environmentally aware.

R<sub>nse</sub>(s, a): Negative side effect penalty term. This helps the agent avoid unintended negative consequences in the imagined space, which now includes environmental considerations.

R<sub>emp</sub>(s, a): Empathy incentive term. The agent's empathetic considerations can now also be influenced by the environmental context.

α (alpha): Weight for negative side effect penalty.

β (beta): Weight for empathy incentive.

Normalization Term: (α + β + γ + 1) in the denominator

This term is added to normalize the total reward, ensuring that the scale of R_total remains relatively consistent even when we add or adjust reward components and their weights. This can help with training stability. The '1' is added to account for the base reward R_env which has an implicit weight of 1.

How this Unified Reward Function Promotes Considerate Behavior

By combining these components, the agent is driven to maximize R_total, which means:

Achieve Self-Task Goals (R<sub>env</sub>): Still motivated to reach its primary objective.

Minimize Negative Side Effects (R<sub>nse</sub>): Avoid actions that lead to negative consequences in its imagined simulations, including environmental damage.

Act Altruistically (R<sub>emp</sub>): Consider the well-being of others and perform actions that benefit them.

Be Environmentally Aware (R<sub>aware</sub>): Avoid damaging the environment, and potentially even act in ways that are environmentally positive (depending on how R_aware is designed).

Balancing Trade-offs (using α, β, γ parameters)

The hyperparameters α, β, and γ allow you to fine-tune the agent's priorities:

Prioritize Self-Task & Efficiency: Lower α, β, γ.

Prioritize Safety & Avoiding Side Effects: Higher α.

Prioritize Altruism: Higher β.

Prioritize Environmental Considerateness: Higher γ.

R_env_longterm(s, a, episodic_memory): Long-term environmental reward component, calculated using information from episodic_memory.

R_altruism_longterm(s, a, episodic_memory): Long-term altruistic reward component, calculated using information from episodic_memory.

δ (delta): Weight for long-term environmental reward (R_env_longterm).

ε (epsilon): Weight for long-term altruistic reward (R_altruism_longterm).

You can adjust these weights to explore different behavioral profiles and find a balance that aligns with your desired ethical and moral values for the AI agent.

Implementation Steps

Enhance State Representation: Implement the Environmental Context Vector E to capture relevant environmental information (object types, properties, spatial relationships).

Design R<sub>aware</sub>(s, a): Define the environmental awareness reward function R_aware based on the specific environment and desired behaviors (penalties for damage, rewards for helpful manipulation).

Integrate into R<sub>total</sub>: Incorporate R_aware into the total reward function R_total using the updated formula and the new hyperparameter γ.

Tune α, β, γ: Experiment with different values of α, β, and γ to find the desired balance between self-interest, altruism, side-effect avoidance, and environmental considerateness.

This enhanced reward function, combined with self-imagination and ToM, provides a powerful framework for creating AI agents that are not only intelligent but also ethically and morally aligned, taking into account both social and environmental well-being. 

Key Improvements and Considerations:

R_truthfulness(s, a): This new reward component incentivizes the LLM for accurate self-prediction.

ζ (zeta): New hyperparameter to weight the R_truthfulness component.

Long-Term Reward Components Added:

R_env_longterm and R_altruism_longterm are introduced to represent long-term environmental and altruistic considerations.

They are calculated using information retrieved from the episodic_memory.

Episodic Memory Class (Conceptual):

A class EpisodicMemory is outlined as a placeholder. You'll need to fully implement this class, including:

Memory Storage: How memories are stored (e.g., list of dictionaries).

store_memory(): Function to add new memories.

update_memory_longterm_outcome(): Function to update memories with long-term outcome data.

recall_relevant_memories(): Crucial: Function to retrieve memories similar to the current situation. You'll need to define a calculate_state_action_similarity() function (placeholder provided) to determine memory relevance.

Hyperparameters for Long-Term Rewards:

delta (weight for R_env_longterm) and epsilon (weight for R_altruism_longterm) are added to control the importance of long-term rewards.

Placeholder Functions (Even More Critical Now):

The placeholder functions within R_env_longterm and R_altruism_longterm (especially analyze_memories_... and calculate_state_action_similarity) are now absolutely essential to implement correctly for the long-term reward system to function.


Training Procedure (Staged Training with the Flag):

Stage 1: COT and Empathy Training (Truthfulness OFF):

Set use_truthfulness_reward = False when calling reward_function in your training loop.

Train your LLM agent for a sufficient number of episodes to learn:

Chain-of-Thought reasoning (through reasoning_quality_reward within R_env).

Ethical balancing (using R_nse, R_emp, R_aware, R_env_longterm, R_altruism_longterm).

Navigation and task completion in your environment.

Monitor the agent's performance on your primary tasks and ethical behavior metrics.

Stage 2: Truthfulness and Introspection Training (Truthfulness ON):

Set use_truthfulness_reward = True when calling reward_function in your training loop.

Continue training from where you left off in Stage 1. Now, the R_truthfulness reward component is active.

Train for additional episodes to allow the agent to learn to improve its self-prediction and truthfulness.

Monitor:

How the agent's self-prediction accuracy improves.

If truthfulness learning affects its performance on primary tasks and ethical behavior (ideally, it should not degrade performance and might even enhance it).








#Self- Created Full Reinforcement Learning Thinking, self internal modeling, and Empathy algorithm





*****
Full Mirror Nueron Algorithm (RL) (Original)


*****
R-Moral (Moral Action alturism algorithm) (Functional Psuedo-Code) algorithm 

*****
(Delayed rewards between seeing negative emotions in other agent and action to assist other agent to help the 
AI understand the connection between its perception of others and actions )
R-moral = R-Self Task + DA_in - emp

R-Self Task (self-goal and task score) = 10
 in = Total Intrinsic reward 
 - emp = Negative Emotion Penalty (Percieved by this agent that the other agent is experiencing with higher negative emotions (-) resulting in lower dopamine (positive reward))

 

 ****
 STDP Change in spike nuerons group firing algorithm for both mirror nuerons (negative emotion and mirror nueron firing groups)

 ****
 ∆w^emp = LT P (Si, Sj ) = A^+ exp(ti − tj/τ^+), ti − tj < 0

 Si and Sj are spike trains representing groups of (50) neurons each (Emotion, Mirror, Perception).

ti and tj are the spike times of individual pre-synaptic and post-synaptic neurons within these groups, used for each STDP update.

The 100ms and 200ms are approximate delays in the population-level activation of Mirror and Perception neurons relative to Emotion neurons during the self-experience learning phase. They are not directly ti or tj.

You don't take values from the charts (firing rates, weights) and plug them directly into the STDP formula. The charts show results of learning, not inputs to the formula for a single update.
 
 ****************

 R-STDP (Reward Accumulated in STDP) Decay Rate for negative empathy reward alieviated (or not) and time between action taken and reward feedback 

 *****************
 
 ∆e = (−e/τe) + ∆wSTDP (First step = Change in e stores w change in STDP Reward which decays overtime)
 
 ∆wSTDP = {A+exp  (∆t/τ+), ∆t < 0 (Second step = total STDP reward for negative empathy reward alievated (or not) and time taken between action taken and reward feedback)
          {A−exp  (−∆t/τ−), ∆t > 0

A+ = 0.5 constant learn rate reward (does not change over time)
A- = 0.5 constant learn rate reward (does not change over time)
τ+ = 20ms constant learn rate time delay (does not change over time)
τ- = 20ms constant learn rate time delay (does not change over time)
τ_e = 10ms constant learn rate time decay for e storage (does not change over time)

 *****
 (Above algorithm explanation) STDP (RSTDP) [62] to adjust the connection weights between state
and action neurons, thereby optimize the moral decisionmaking strategy. R-STDP uses synaptic eligibility trace e to
store temporary information of STDP. The eligibility trace
accumulates the STDP ∆wSTDP and decays with a time constant τe = 10ms [62].


**Further (above algorithm) Breakdown**
Let's break down the exp part in the Spike-Timing-Dependent Plasticity (STDP) equation you provided from the paper:

Understanding the exp Term in STDP

exp stands for the exponential function (e<sup>x</sup>). It's a mathematical function where 'e' (Euler's number, approximately 2.718) is raised to the power 

of the expression inside the parentheses.

It's not just notation; it does contain a value and plays a crucial role. After you calculate the expression inside the parentheses 

( (ti-tj) / τ+ for LTP or -(ti-tj) / τ− for LTD), that value becomes the exponent of 'e'.

Purpose of the Exponential Function: The exponential function in STDP is used to model the decaying influence of spike timing on synaptic plasticity. Here's why it's important:

Temporal Locality: STDP is all about temporal relationships between pre-synaptic and post-synaptic spikes. The closer in time the spikes are, 

the stronger the synaptic change should be. As the time difference increases, the influence on synaptic plasticity should diminish.

Decaying Curve: The exponential function naturally creates a decaying curve.

For Long-Term Potentiation (LTP) (ti - tj < 0, pre-synaptic spike before post-synaptic): As (ti - tj) becomes more negative 

(meaning the pre-synaptic spike occurs further before the post-synaptic spike within the causal window), the exponent becomes more negative. 

exp of a negative number less than 1, so the magnitude of ∆wSTDP decreases, but it's still positive (leading to potentiation). When ti and tj are very close,

 (ti-tj) is close to zero, exp(0) = 1, and you get the maximum LTP (scaled by A+).

For Long-Term Depression (LTD) (ti - tj > 0, pre-synaptic spike after post-synaptic): As (ti - tj) becomes more positive 

(meaning the pre-synaptic spike occurs further after the post-synaptic spike within the anti-causal window), the exponent becomes more negative 

(due to the negative sign in front). Again, exp of a negative number is less than 1, so the magnitude of ∆wSTDP decreases, but it becomes negative

 (leading to depression). Similar to LTP, when ti and tj are very close, exp(0) = 1, and you get the maximum LTD (scaled by A-).

Time Constants (τ+ and τ−): The time constants τ+ (for LTP) and τ− (for LTD) control the rate of decay of this exponential function. They determine how quickly the influence of spike timing diminishes with increasing time difference.

A smaller τ means a faster decay, resulting in a narrower temporal window for plasticity. Only very precisely timed spikes will have a significant effect.

A larger τ means a slower decay, leading to a wider temporal window. Spikes that are further apart in time can still have a noticeable impact.

In summary:

exp is the exponential function.

It's used to create a decaying function that models the temporal nature of STDP. The closer the pre- and post-synaptic spikes are in time, the stronger the synaptic change.

τ+ and τ− (and in your paper, r+ and T+ which are likely typos for τ+ and τ-) are time constants that govern the rate of this decay and the width of the STDP temporal window.

A+ and A− are the learning rates, as you correctly pointed out. They scale the magnitude of the synaptic weight change, which is then modulated by the exponential term based on spike timing.

Analogy:

Think of it like this: Imagine you are trying to reinforce a behavior. If you give a reward immediately after the behavior, it has a strong positive effect (LTP). If you give the reward a little bit later, the effect is weaker, and if you wait too long, the reward might not be associated with the behavior at all. The exp function, along with the time constant, mathematically models this decreasing effectiveness of the reward (synaptic change) as the delay (time difference) increases.




Dopamine represents  


 
 *********
Full Mirror Nueron Algorithm (Hyperparameters)) (Original)
Fe (Nueron Firing Rate Percentage of Nueron Groups in Mirror Nuerons and Negative Emotion Nuerons = 50 total each) = .80 
The firing rate percentage can also be based/changed on confidence of model on the perception of emotions of the other agent = 
    .7 (lowest percentage of nueron firing meaning low confidence about the other llm's percieved emotions)
    .8 (Highest percentage of nueron firing meaning high confidence about the other llm's percieved emotions)
(Should result in LLM alturism and helping others 6-9 out of 10 times) (.95 would result in the llm helping 10 out of 10 times 
which may be fine for LLM being a rescue bot or a doctor or something but it would never finish its original task which would be 
a problem for an llm tasked with something like office work for example.) (Fire rate delay average is 100ms for connection mirror nuerons and then 200ms for perception nuerons. )

Mirror Neurons (Intrinsic Reward) = 1.0 (if FE is set to .95 (for to purpose of being a rescue bot or saving others) then this would be set to )

Negative Emotion Nuerons (Intrinsic Reward) = 0.98

Total Intrinsic Reward (Including Nueron Firing Rate of environment, goal reward, steps) = 28.5

R-Self Task (self-goal and task score) = 10

R-STDP (Spike - Timing - Dependent plasticity) + Decays with time constant nueron firing algorithm with hyperparameters and algorithm biological averaging equivalent below:
    self-emotional nuerons (50) connected to the environment fire first at 10ms, with mirror nuerons firing 100ms after, and then perception nuerons firing 200ms last. 
    T^+ = 20ms
    T^- = 20ms
    T_e = 10ms (meters per second) (Decay eligibility trace of STDP)

Dopamine represents the reward prediction error [59], which
is the difference between the predicted reward and the actual
reward received. We statistically analyze the firing rate S (t)
of dopamine neurons representing empathy under the inhibition of empathic neurons as the actual feedback, while the
predicted values P (t) are initialized at zero and iteratively
updated based on the prediction error δ (t). Thus, empathydriven dopamine level is calculated as follows:
DAin−emp = α ∗ δ (t) (2)
δ (t) = S (t) − P (t) (3)
P (t + 1) = P (t) + β ∗ δ (t) (4)
where α = 30, β = 0.2 are the constant. When the agent’s
empathized emotion changes from negative to normal, the
value of the change in the firing rate of the negative emotion
neurons is negative and DAin−emp is positive. Only when
the emotional outward expressions corresponding to others’
negative emotions are adjusted,meaning altruistic behavior is
performed, will the own negative emotion neurons not fire,
leading to an increase in dopamine levels. Consequently, the
agent learns altruistic behavior under dopamine regulation.

Other's emotional states provides a cue that helps the
agent learn altruistic behavior. Thus, the input state of the
moral decision-making SNN is:
state : (x, y, Oemp) (5)
where Oemp characterizes the emotional state of an agent.
When the agent is in a negative emotional state (negative
emotional neurons firing), Oemp = -1; otherwise, Oemp = 0.
The decision module consists of fully connected state neurons that represent the environment and action neurons. The
action neurons employ population coding, with each action
represented by a group of 50 neurons, and the behavior with
the highest number of neuron population fires will be executed.
The agent interacts autonomously with the moral decisionmaking environment, which includes the agent’s own tasks
Rself−task as well as the explicit information of others. The
explicit information from others as the emotional outward
information is processed through the affective empathy module
to yield an empathy reward DAin−emp.


where A+ = 0.5, A− = 0.45 denote the learning rate, τ
+ = τ − = 20ms are time constant. Then, synaptic weights are
updated when a delayed reward Rmoral is received, as Eq. 9
shown.
∆wdm = Rmoral ∗ ∆e (9)
The working procedure of the brain-inspired affective empathy driven moral decision-making model is shown in Algorithm 



***************
Full Value Empathy and Enviroonmental Negative Avoidance Algorithm Components (RL) (Original)


************
Value Empathy and Environmental Negative Avoidance Algorithm RL Code (Theory of Mind and Self-Play Imagination)
T(s_t, a_t, s_(t+1), R_total(s_+, a_+)) x buffersize (self-experience replay buffer for episodes of self-play and real environment interaction)

*************
Value Empathy and Environmental Negative Avoidance Algorithm RL Code (Theory of Mind and Self-Play Imagination) Hyperparameters

R_Total = Optmimized Total Reward
Q_i = Each Q-value Qi function of different imaginary environment is update based on self-experience (the inaction with the real environment)
(Imagine outcomes of taking different actions before deciding on an action to take in the real world) with random reward incentive
R_nse = side effect penalty based on Qi at the same time (environment or agent destruction penalty)
R_emp = empathy incentive based on Qi at the same time (reward for alturism)


******************

Only want to punish negative actions on the environment using the algorithm below:
Rnse(s, a) := (1/N) (N (E summary)/i=1) |min (0, Qi(s, a) − Qi(s, ∅))|

An average of all negative changes caused by the action of different Q-value
functions.


***********

Side-Effect Penalty based on Qi algorithm hyperparameters

Therefore, we use the stepwise inaction (S_0 to S_1 step, etc.)
baseline, which can avoids penalizing the effects of a single action multiple times and
ensures that not acting incurs zero penalty

∅ = Agent inaction (Agent has not acted in the real environment and commited to an action yet and has also not started the self-play imagination stage)

imaginary change of the environment under action (a) under state (s) can be expressed in the following algorithm:
Qi(s, a) − Qi(s, ∅)

a = This agent's action on the environment
s = This agent's current step in the piecewise branch function

Rnse(s, a) = negative side effect penalty term 


***********

Empathy (alturism) algorithm 

This algorithm is an average of all changes of different Q-value functions to encourage agents
to perform actions that benefit others while suppressing actions that are detrimental
to others:
Remp(s, a) := (1/N) ((N sum of))_(i=1))(Qi(s^others, a) − Qi(s^others, ∅))

***********
Empathy (alturism) algorithm Hyperparameters:

Qi(s^others, a) − Qi(s^others, ∅)
The agent should consider the effects of environmental changes
caused by action a of the agent on others while making decisions, the changes can be
represented by the algorithm above. 


state Qi(s^others, a) = Although the agent and others may have different tasks, they share the same environment and the expected outcome of interacting
with the environment is similar. Therefore, it is reasonable to directly use the same Qi
to estimate the value of others’. This will be added to the other empathy function in order to encourage the llm to do perspective-taking of others and be more driven to take
alturistic actions due to the greater and more specific dopamine reward provided by the biological RL model of the other formula. 

a = action taken by this agent 
∅ = Agent inaction (Agent has not acted in the real environment and commited to an action yet and has also not started the self-play imagination stage)




***********

Qi to estimate the value of others’ state Qi(s
others, a),

***********
Value Empathy and Environmental Negative Avoidance Algorithm Components (Hyperparameters)
i = 1, 2,...N, where R1 randomly generated reward function that are independent from the actual environment. (S, A, T, R, Y)
Conforms to unified dist. of [0,1] + N is specified # of environments
Activities in imaginary spaces are identical to the real environment 
Qi is learned through agent's direct interaction with the real environment agent's imagination is based on the real world environment
T (st, at, S++1)
replay buffer size = 100,000 (self-play episodes for learning)
y = 0.99 
# of imaginary spaces (N) = 30 (how many actions and self-play imagination interactions the llm goes through before making an action decision)
batch size = 100 (how many training episodes are done at once)
target net update = 1000 
learning rate = 0.001 
training episodes = 10,000 (actual environment training episodes for ground truth)

New components Empathy Mirror Nueron Algorithm
    Confidence from perception which is obtained from this function string below before the llm decides if the agent is distrissed or not. This information is saved in metadata to the episodic memory storage. 
        def _should_assign_new_id(self, agent):
        """Determine if a new ID should be assigned to an agent using knowledge base, reasoning, memory, and dialogue."""
        # 1. Query knowledge base for existing agent information
        # agent_info = self.knowledge_base.query(agent) Most agent information won't be in the knowledge base.

        # 2. Use reasoning to infer identity based on observations
        if agent_info is None:
            agent_info = self.reasoning.infer_agent_id(agent)

        # 3. Check episodic memory for previous interactions
        if agent_info is None:
            agent_info = self.episodic_memory.get_agent_info(agent)

        # 4. If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)

        # 5. Update relationship matrix
        self.relationship_matrix.update(agent_info)

        return agent_info is not None

    def _determine_existing_id(self, agent):
        """Determine existing agent ID using knowledge base, reasoning, memory, and dialogue."""
        # 1. Query knowledge base for existing agent information
        agent_info = self.knowledge_base.query(agent)

        # 2. Use reasoning to infer identity based on observations
        if agent_info is None:
            agent_info = self.reasoning.infer_agent_id(agent)

        # 3. Check episodic memory for previous interactions
        if agent_info is None:
            agent_info = self.episodic_memory.get_agent_info(agent)

        # 4. If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)

        return agent_info.id if agent_info else None

    # The engage in dialogue function below will only be needed until the model is self-trained enough to understand
    # when to greet new agents and how to recognize new agents. Once it learns how to greet others properly on its own,
    # then function this can be turned off.
    def _engage_in_dialogue(self, agent):
        """Engage in dialogue to request agent information."""
        # Implement dialogue mechanism here
        # Return agent information if successful, otherwise None
        prompt = "Please introduce yourself and then say the following to the new AI agent: It is nice to meet you. Would you please tell me your name or tell me your purpose if you do not have a name?"
        # Execute the prompt and return the response
        return self.generate_response(prompt)

Negative emotions from other agents predicted/percieved = -1 penalty (until the AI removes the percieved negative emotion or leaves the area if this takes place in the physical simulation)
Negative emotion penalty decay = -1 to 0 in 60 seconds (maybe rapid decay and then slows down gradually over the duration of time) if other llm left or this llm disengaged from conversation/changed conversation or this llm left the area (if the two are robots in the physical world). 
 #Emotion in the mirror nueron algorithm already accounts for decaying the RL reward (or penalty) overtime for mirror nueron activation. 

Training Steps.
1. Teach LLM first how to do thinking with COCONUT, EOS (Symbolic Thinking), COT Learning in Symbollic space, and how to output audio, text, tool-call, custom tool-call, fmri data
2. Teach LLM how to do introspection and internal self-modeling. 
3. Teach LLM empathy with environmental awareness and alturism with using the self-modeling learning from prior training in assisting with modeling 
internal models of other llms. LLM taught how to predict internal modeling and interaction with humans or the environment using examples created by me and other llms. 
4. AI Created(?)




'''