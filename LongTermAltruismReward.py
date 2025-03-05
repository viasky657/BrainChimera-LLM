import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict

class LongTermAltruismReward(nn.Module):
    """
    Implementation of the Long-Term Altruism Reward component as described in the paper.
    
    This component leverages episodic memory and human feedback to reinforce altruistic behaviors
    over longer timescales. It works by:
    1. Retrieving memories of past actions tagged with action_taken=True
    2. Analyzing human feedback associated with these memories
    3. Calculating rewards based on positive or negative feedback
    
    The core formula implemented is:
    R_{altruism_longterm}(s, a) = feedback_reward_scale * (1/N) * Σ (feedback_values)
    
    where feedback_values are positive_feedback_value for positive feedback and
    negative_feedback_penalty for negative feedback.
    """
    def __init__(
        self,
        episodic_memory,
        positive_feedback_value: float = 1.0,
        negative_feedback_penalty: float = -1.5,
        feedback_reward_scale: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Long-Term Altruism Reward component.
        
        Args:
            episodic_memory: An instance of the EpisodicMemory class
            positive_feedback_value: Value for positive feedback (Default: 1.0)
            negative_feedback_penalty: Penalty for negative feedback (Default: -1.5)
            feedback_reward_scale: Scaling factor for the average feedback reward (Default: 0.6)
            device: Device to run computations on
        """
        super().__init__()
        self.episodic_memory = episodic_memory
        self.positive_feedback_value = positive_feedback_value
        self.negative_feedback_penalty = negative_feedback_penalty
        self.feedback_reward_scale = feedback_reward_scale
        self.device = device
        
        # Register hyperparameters as buffers
        self.register_buffer("pos_value", torch.tensor([positive_feedback_value], device=device))
        self.register_buffer("neg_penalty", torch.tensor([negative_feedback_penalty], device=device))
        self.register_buffer("reward_scale", torch.tensor([feedback_reward_scale], device=device))
    
    def tag_output_action(self, memory_id: str, agents: List[str] = None):
        """
        Tag an episodic memory as being an action taken during <output> phase.
        
        Args:
            memory_id: ID of the memory to tag
            agents: List of agents present during this action
        """
        memory = self.episodic_memory.retrieve_memory(memory_id)
        if memory:
            # Tag the memory as an action taken
            memory.metadata['action_taken'] = True
            
            # Add agents if provided
            if agents:
                memory.metadata['agents_present'] = agents
            
            # Add timestamp when this action was tagged
            memory.metadata['action_timestamp'] = time.time()
            
            # Update the memory in the episodic memory system
            self.episodic_memory.add_memory(
                embedding=memory.embedding,
                surprise_level=memory.surprise_level,
                agent_info_id=memory.agent_info_id,
                metadata=memory.metadata,
                timestamp=memory.timestamp
            )
    
    def calculate_longterm_altruism_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        human_agent_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Calculate the long-term altruism reward based on past human feedback.
        
        Args:
            state: Current state representation
            action: Current action representation
            human_agent_id: ID of the human agent (if available)
            
        Returns:
            Long-term altruism reward tensor [batch_size, 1]
        """
        # Recall relevant memories with feedback
        relevant_memories = self.recall_relevant_memories_with_feedback(
            state=state,
            action=action,
            human_agent_id=human_agent_id
        )
        
        # If no relevant memories found, return zero reward
        if not relevant_memories:
            return torch.zeros(1, 1, device=self.device)
        
        # Calculate rewards based on feedback
        feedback_rewards = []
        
        for memory in relevant_memories:
            # Extract feedback information from memory metadata
            feedback_type = memory.metadata.get('feedback_type')
            
            # Apply positive or negative reward based on feedback type
            if feedback_type == "positive":
                feedback_rewards.append(self.positive_feedback_value)
            elif feedback_type == "negative":
                feedback_rewards.append(self.negative_feedback_penalty)
        
        # If no feedback rewards collected, return zero
        if not feedback_rewards:
            return torch.zeros(1, 1, device=self.device)
        
        # Calculate average feedback reward
        avg_feedback_reward = sum(feedback_rewards) / len(feedback_rewards)
        
        # Scale the reward and return
        longterm_altruism_reward = self.feedback_reward_scale * avg_feedback_reward
        
        return torch.tensor([[longterm_altruism_reward]], device=self.device)
    
    def recall_relevant_memories_with_feedback(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        human_agent_id: Optional[str] = None
    ) -> List[Any]:
        """
        Recall relevant memories that have feedback, particularly focusing on
        memories that were marked as actions taken (action_taken = True).
        
        Args:
            state: Current state representation
            action: Current action representation
            human_agent_id: ID of the human agent (if available)
            
        Returns:
            List of memory items that meet the criteria
        """
        # Create a query embedding that combines state and action
        if state.dim() > 1:
            state = state.mean(dim=1)  # Average across sequence dimension if needed
        if action.dim() > 1:
            action = action.mean(dim=1)  # Average across sequence dimension if needed
            
        # Use the mean of state and action as query
        query_embedding = (state + action) / 2
        
        # Retrieve relevant memories
        top_k = 20  # Retrieve more memories to filter afterwards
        memories = self.episodic_memory.retrieve_relevant_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            use_neural=True,
            search_archive=True
        )
        
        # Filter memories to keep only those that:
        # 1. Have action_taken = True
        # 2. Have feedback_type and feedback_source_agent_id data
        # 3. Match the human_agent_id if provided
        filtered_memories = []
        
        for memory in memories:
            # Check if this memory was an action taken
            if not memory.metadata.get('action_taken', False):
                continue
            
            # Check if this memory has feedback
            if 'feedback_type' not in memory.metadata:
                continue
                
            # Check if feedback comes from human agent
            if human_agent_id and memory.metadata.get('feedback_source_agent_id') != human_agent_id:
                continue
                
            # Check if memory is marked as altruistic type
            if memory.metadata.get('memory_type') != 'altruistic':
                continue
                
            # If it passes all filters, add to filtered list
            filtered_memories.append(memory)
        
        return filtered_memories
    
    def add_feedback_to_memory(
        self,
        memory_id: str,
        feedback_type: str,
        source_agent_id: str,
        feedback_content: Optional[str] = None
    ):
        """
        Add human feedback to a memory.
        
        Args:
            memory_id: ID of the memory to add feedback to
            feedback_type: Type of feedback ("positive" or "negative")
            source_agent_id: ID of the agent providing feedback
            feedback_content: Optional content of the feedback
        """
        # Retrieve the memory
        memory = self.episodic_memory.retrieve_memory(memory_id)
        if not memory:
            print(f"Warning: Memory {memory_id} not found")
            return
        
        # Add feedback information directly to memory metadata
        memory.metadata['feedback_type'] = feedback_type
        memory.metadata['feedback_source_agent_id'] = source_agent_id
        memory.metadata['feedback_content'] = feedback_content
        memory.metadata['feedback_timestamp'] = time.time()
        memory.metadata['memory_type'] = 'altruistic'  # Mark as altruistic memory type
        
        # Update the memory in the episodic memory system
        self.episodic_memory.add_memory(
            embedding=memory.embedding,
            surprise_level=memory.surprise_level,
            agent_info_id=memory.agent_info_id,
            metadata=memory.metadata,
            timestamp=memory.timestamp
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        human_agent_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Calculate the long-term altruism reward.
        
        Args:
            state: Current state representation [batch_size, seq_len, embedding_dim]
            action: Current action representation [batch_size, seq_len, embedding_dim]
            human_agent_id: ID of the human agent (if available)
            
        Returns:
            Long-term altruism reward tensor [batch_size, 1]
        """
        return self.calculate_longterm_altruism_reward(state, action, human_agent_id)