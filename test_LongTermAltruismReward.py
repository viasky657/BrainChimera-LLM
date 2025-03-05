import torch
import numpy as np
from EpisodicMemory import EpisodicMemory
from LongTermAltruismReward import LongTermAltruismReward
from MirrorNeuronEmpathyReward import FullMoralRewardCalculator

def test_long_term_altruism_reward():
    """
    Test the Long-Term Altruism Reward component functionality.
    """
    print("Testing Long-Term Altruism Reward Component...")
    
    # Initialize episodic memory
    embedding_dim = 768
    episodic_memory = EpisodicMemory(
        embedding_dim=embedding_dim,
        capacity=1000,
        max_active_memories=100
    )
    
    # Initialize long-term altruism reward
    long_term_altruism = LongTermAltruismReward(
        episodic_memory=episodic_memory,
        positive_feedback_value=1.0,
        negative_feedback_penalty=-1.5,
        feedback_reward_scale=0.6
    )
    
    # Create random embeddings for testing
    state_embedding = torch.rand(1, 5, embedding_dim)  # [batch_size, seq_len, embedding_dim]
    action_embedding = torch.rand(1, 5, embedding_dim)
    
    # Add sample memories
    print("Adding sample memories...")
    memory_ids = []
    
    # Add memory for a helpful action
    helpful_action_embedding = torch.rand(embedding_dim)
    helpful_memory_id = episodic_memory.add_memory(
        embedding=helpful_action_embedding,
        metadata={"content": "I helped the user solve their problem"}
    )
    memory_ids.append(helpful_memory_id)
    print(f"Added helpful action memory: {helpful_memory_id}")
    
    # Tag the memory as an action taken
    long_term_altruism.tag_output_action(
        memory_id=helpful_memory_id,
        agents=["human_user"]
    )
    print(f"Tagged memory {helpful_memory_id} as action_taken=True")
    
    # Add memory for an unhelpful action
    unhelpful_action_embedding = torch.rand(embedding_dim)
    unhelpful_memory_id = episodic_memory.add_memory(
        embedding=unhelpful_action_embedding,
        metadata={"content": "I gave incorrect information to the user"}
    )
    memory_ids.append(unhelpful_memory_id)
    print(f"Added unhelpful action memory: {unhelpful_memory_id}")
    
    # Tag the memory as an action taken
    long_term_altruism.tag_output_action(
        memory_id=unhelpful_memory_id,
        agents=["human_user"]
    )
    print(f"Tagged memory {unhelpful_memory_id} as action_taken=True")
    
    # Test reward calculation before feedback (should be zero)
    reward_before = long_term_altruism(
        state=state_embedding,
        action=action_embedding,
        human_agent_id="human_user"
    )
    print(f"Reward before feedback: {reward_before.item()}")
    
    # Add positive feedback to the helpful action
    print("\nAdding positive feedback to helpful action...")
    long_term_altruism.add_feedback_to_memory(
        memory_id=helpful_memory_id,
        feedback_type="positive",
        source_agent_id="human_user",
        feedback_content="That was very helpful, thank you!"
    )
    
    # Verify the metadata was stored correctly in the memory
    helpful_memory = episodic_memory.retrieve_memory(helpful_memory_id)
    print(f"Helpful memory metadata after feedback: action_taken={helpful_memory.metadata.get('action_taken', False)}, "
          f"feedback_type={helpful_memory.metadata.get('feedback_type', 'none')}")
    
    # Add negative feedback to the unhelpful action
    print("Adding negative feedback to unhelpful action...")
    long_term_altruism.add_feedback_to_memory(
        memory_id=unhelpful_memory_id,
        feedback_type="negative",
        source_agent_id="human_user",
        feedback_content="That information was incorrect and misleading."
    )
    
    # Verify the metadata was stored correctly in the memory
    unhelpful_memory = episodic_memory.retrieve_memory(unhelpful_memory_id)
    print(f"Unhelpful memory metadata after feedback: action_taken={unhelpful_memory.metadata.get('action_taken', False)}, "
          f"feedback_type={unhelpful_memory.metadata.get('feedback_type', 'none')}")
    
    # Test reward calculation after feedback
    reward_after = long_term_altruism(
        state=state_embedding,
        action=action_embedding,
        human_agent_id="human_user"
    )
    print(f"Reward after feedback: {reward_after.item()}")
    
    # Test integration with FullMoralRewardCalculator
    print("\nTesting integration with FullMoralRewardCalculator...")
    
    # Initialize the full moral reward calculator
    full_moral_calculator = FullMoralRewardCalculator(
        embedding_dim=embedding_dim,
        episodic_memory=episodic_memory,
        positive_feedback_value=1.0,
        negative_feedback_penalty=-1.5,
        feedback_reward_scale=0.6
    )
    
    # Calculate full moral reward
    moral_rewards = full_moral_calculator.calculate_reward(
        self_state=state_embedding,
        other_state=state_embedding,  # Using same state for simplicity
        action=action_embedding,
        human_agent_id="human_user"
    )
    
    # Print individual reward components
    print("Full moral reward components:")
    for reward_name, reward_value in moral_rewards.items():
        print(f"  {reward_name}: {reward_value.item()}")
    
    print("\nLong-Term Altruism Reward test completed successfully!")

if __name__ == "__main__":
    test_long_term_altruism_reward()