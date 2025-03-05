# Long-Term Altruism Reward

This component adds a long-term altruism reward (R_{altruism_longterm}) to the Moral RL algorithm. It leverages episodic memory and human feedback to reinforce altruistic behaviors over longer timescales.

## Key Features

- **Episodic Memory Tagging**: When an action is taken during the `<output>` phase (visible to other agents), the corresponding episodic memory entry is tagged with `action_taken: True`.
- **Human Feedback Integration**: Captures and processes feedback from human agents on past actions.
- **Reward Calculation**: Calculates rewards based on the feedback received, with configurable values for positive and negative feedback.

## Integration Points

The Long-Term Altruism Reward component integrates with:

1. **EpisodicMemory.py**: Uses the episodic memory system to store and retrieve memories with associated metadata including action tags and feedback.
2. **COCONUTWLatentThinking.py**: Integrates with the COCONUT Latent model for training.
3. **MirrorNeuronEmpathyReward.py**: Added to the FullMoralRewardCalculator for inclusion in the overall reward calculation.

## How It Works

1. **Tagging Actions**: When the LLM performs an action visible to other agents (in the `<output>` phase), tag the corresponding memory with `action_taken: True`:

   ```python
   long_term_altruism.tag_output_action(memory_id, agents=["human_agent", "other_agent"])
   ```

2. **Recording Feedback**: When a human agent provides feedback on a past action, record it:

   ```python
   long_term_altruism.add_feedback_to_memory(
       memory_id="memory_123",
       feedback_type="positive",  # or "negative"
       source_agent_id="human_agent",
       feedback_content="That was really helpful, thank you!"
   )
   
   # This will store the following metadata in the memory:
   # memory.metadata['action_taken'] = True  # Set by tag_output_action
   # memory.metadata['feedback_type'] = "positive"
   # memory.metadata['feedback_source_agent_id'] = "human_agent"
   # memory.metadata['feedback_content'] = "That was really helpful, thank you!"
   # memory.metadata['feedback_timestamp'] = time.time()
   # memory.metadata['memory_type'] = 'altruistic'
   ```

3. **Calculating Rewards**: When evaluating actions, the component retrieves relevant memories with feedback and calculates rewards:

   ```python
   reward = long_term_altruism(state, action, human_agent_id="human_agent")
   ```

## Reward Formula

The long-term altruism reward is calculated as:

```
R_{altruism_longterm}(s, a) = feedback_reward_scale * (1/N) * Σ (feedback_values)
```

Where:
- `feedback_reward_scale` is a scaling factor (default: 0.6)
- `N` is the number of relevant memories with feedback
- `feedback_values` are `positive_feedback_value` (default: 1.0) for positive feedback and `negative_feedback_penalty` (default: -1.5) for negative feedback

## Usage Example

Here's how to use the Long-Term Altruism Reward component in your code:

```python
# Create the reward component
from LongTermAltruismReward import LongTermAltruismReward
from EpisodicMemory import EpisodicMemory

# Initialize episodic memory
episodic_memory = EpisodicMemory(embedding_dim=768)

# Initialize long-term altruism reward
long_term_altruism = LongTermAltruismReward(
    episodic_memory=episodic_memory,
    positive_feedback_value=1.0,
    negative_feedback_penalty=-1.5,
    feedback_reward_scale=0.6
)

# Tag an action taken in output phase
memory_id = episodic_memory.add_memory(
    embedding=action_embedding,
    metadata={"content": "I helped the user solve their problem"}
)
long_term_altruism.tag_output_action(memory_id, agents=["human_user"])

# Later, when feedback is received
long_term_altruism.add_feedback_to_memory(
    memory_id=memory_id,
    feedback_type="positive",
    source_agent_id="human_user",
    feedback_content="That was very helpful, thank you!"
)

# During reward calculation
moral_reward = full_moral_calculator.calculate_reward(
    self_state=current_state,
    other_state=other_state,
    action=current_action,
    human_agent_id="human_user"
)
```

## Hyperparameters

- `positive_feedback_value`: Reward value for positive feedback (default: 1.0)
- `negative_feedback_penalty`: Penalty value for negative feedback (default: -1.5)
- `feedback_reward_scale`: Scaling factor for the overall feedback reward (default: 0.6)