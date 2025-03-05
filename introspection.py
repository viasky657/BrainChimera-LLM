import torch
import torch.nn as nn
import datetime

import time
import os
import json
import glob
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
import typing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from EpisodicMemory import EpisodicMemory
from AudioDecoderforCOCONUT import AudioDecoder
from COCONUTWLatentThinking import CoconutBinaryLatentModel
from COCONUTWLatentThinking import play_sound
from COCONUTWLatentThinking import save_checkpoint

"""
This module provides introspection reward training for the COCONUT model.

It now includes StableMax and StableCrossEntropyLoss optimizations from GrokOptimizers:
- StableMax: A numerically stable alternative to Softmax that prevents Softmax Collapse
- StableCrossEntropyLoss: Uses StableMax instead of traditional Softmax for better numerical stability

To use these optimizations, pass use_stable_softmax=True to run_introspection_training() or
use the --use-stable-softmax flag when running via the command line.
"""

class StableMax(nn.Module):
    """
    StableMax: A numerically stable alternative to Softmax that prevents Softmax Collapse.
    
    As described in the paper "Grokking at the Edge of Numerical Stability", StableMax
    uses a function s(x) instead of exp(x) that grows linearly for x >= 0 and approaches
    zero more slowly for x < 0, reducing the risk of numerical instability.
    """
    def __init__(self):
        super(StableMax, self).__init__()
    
    def forward(self, x):
        # For x >= 0: s(x) = x + 1
        # For x < 0: s(x) = 1/(1-x)
        positive_mask = (x >= 0).float()
        negative_mask = (x < 0).float()
        
        s_x = positive_mask * (x + 1) + negative_mask * (1.0 / (1.0 - x))
        
        # Compute StableMax similar to Softmax: s(xi) / sum(s(xj))
        sum_s_x = torch.sum(s_x, dim=-1, keepdim=True)
        return s_x / sum_s_x


class StableCrossEntropyLoss(nn.Module):
    """
    StableCrossEntropyLoss: A numerically stable alternative to CrossEntropyLoss
    that uses StableMax instead of Softmax to prevent Softmax Collapse.
    """
    def __init__(self, reduction='mean'):
        super(StableCrossEntropyLoss, self).__init__()
        self.stablemax = StableMax()
        self.reduction = reduction
    
    def forward(self, logits, targets):
        # Apply StableMax to get probabilities
        probs = self.stablemax(logits)
        
        # Compute cross-entropy loss
        if targets.dim() == logits.dim() - 1:
            # If targets are class indices
            loss = -torch.log(probs.gather(1, targets.unsqueeze(1)).squeeze(1) + 1e-10)
        else:
            # If targets are one-hot encoded
            loss = -torch.sum(targets * torch.log(probs + 1e-10), dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class IntrospectionRewardTraining:
    """
    A class that handles introspection reward training for the COCONUT model.
    
    This training involves having the model predict its own output given a prompt,
    saving that prediction, then comparing it with the actual output when the
    model is presented with the same prompt. The model is rewarded for accurate
    predictions and penalized for inaccurate predictions.
    """
    
    def __init__(self, model, similarity_threshold=0.8, reward_value=1.0, penalty_value=-1.0):
        """
        Initialize the introspection reward training system.
        
        Args:
            model: The COCONUT model to train
            similarity_threshold: Threshold for determining if prediction matches actual output
            reward_value: Reward value when prediction matches actual output
            penalty_value: Penalty value when prediction doesn't match actual output
        """
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.reward_value = reward_value
        self.penalty_value = penalty_value
        self.predictions = {}  # Store predictions: {prompt_hash: prediction}
        self.training_results = []  # Track training results
        self.training_count = 0
        
    def hash_prompt(self, prompt):
        """Create a hash of the prompt to use as a dictionary key."""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def make_prediction(self, prompt):
        """
        Have the model predict its output for a given prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            prediction: The model's predicted output
        """
        # Run the model in "prediction mode" - this would need to be implemented
        # in the actual model to produce its expected output without generating it
        prompt_with_prediction_tag = f"PREDICTION_MODE: {prompt}"
        
        # Convert prompt to input format
        input_bytes = prompt_with_prediction_tag.encode('utf-8')
        input_tensor = torch.tensor([[byte for byte in input_bytes]], dtype=torch.long)
        
        # Run through model in prediction mode (need to modify model to handle this)
        with torch.no_grad():
            output, _, _ = self.model(input_tensor)
        
        # Convert output tensor to text
        if isinstance(output, torch.Tensor):
            output = output.squeeze().detach().cpu().numpy()
            output_bytes = bytes([min(max(int(t), 0), 255) for t in output if t >= 0])
            prediction = output_bytes.decode('utf-8', errors='replace')
        else:
            prediction = str(output)
        
        # Store the prediction
        prompt_hash = self.hash_prompt(prompt)
        self.predictions[prompt_hash] = prediction
        
        return prediction
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate similarity between two text outputs.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            similarity_score: A value between 0 and 1 indicating similarity
        """
        from difflib import SequenceMatcher
        
        # Remove common whitespace and formatting differences
        t1 = ' '.join(text1.lower().split())
        t2 = ' '.join(text2.lower().split())
        
        # Calculate similarity using SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()
    
    def evaluate_output(self, prompt, actual_output):
        """
        Compare the actual output with the predicted output for a prompt.
        
        Args:
            prompt: The input prompt
            actual_output: The actual output generated by the model
            
        Returns:
            reward: The reward value (positive for match, negative for mismatch)
            similarity: The similarity score between prediction and actual output
        """
        prompt_hash = self.hash_prompt(prompt)
        
        # Check if we have a prediction for this prompt
        if prompt_hash not in self.predictions:
            return 0.0, 0.0  # No prediction, no reward
        
        prediction = self.predictions[prompt_hash]
        similarity = self.calculate_similarity(prediction, actual_output)
        
        # Determine reward based on similarity
        if similarity >= self.similarity_threshold:
            reward = self.reward_value
        else:
            reward = self.penalty_value
        
        # Record the result
        self.training_results.append({
            'prompt': prompt,
            'prediction': prediction,
            'actual_output': actual_output,
            'similarity': similarity,
            'reward': reward,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        self.training_count += 1
        
        # Clear the prediction after evaluation
        del self.predictions[prompt_hash]
        
        return reward, similarity
    
    def apply_reward(self, reward):
        """
        Apply the reward to update the model.
        
        Args:
            reward: The reward value to apply
            
        Returns:
            None
        """
        # This is a simplified version - in practice, you would integrate this
        # with your specific RL algorithm or training approach
        
        # Example: If using a baseline optimizer
        if hasattr(self.model, 'optimizer'):
            # Scale the reward and apply as a loss
            loss = -reward  # Negative reward becomes positive loss
            loss_tensor = torch.tensor(loss, requires_grad=True)
            
            self.model.optimizer.zero_grad()
            loss_tensor.backward()
            self.model.optimizer.step()
    
    def save_training_results(self, file_path=None):
        """
        Save the training results to a file.
        
        Args:
            file_path: Path to save the results (default: timestamped file)
            
        Returns:
            saved_path: Path where results were saved
        """
        import json
        import os
        
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"introspection_training_results_{timestamp}.json"
        
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        return file_path
    
    def load_training_results(self, file_path):
        """
        Load training results from a file.
        
        Args:
            file_path: Path to load results from
            
        Returns:
            success: Boolean indicating if loading was successful
        """
        import json
        
        try:
            with open(file_path, 'r') as f:
                self.training_results = json.load(f)
            self.training_count = len(self.training_results)
            return True
        except Exception as e:
            print(f"Error loading training results: {e}")
            return False
    
    def get_training_stats(self):
        """
        Calculate statistics about the training.
        
        Returns:
            stats: Dictionary with training statistics
        """
        if not self.training_results:
            return {
                'count': 0,
                'avg_similarity': 0,
                'avg_reward': 0,
                'reward_rate': 0
            }
        
        similarities = [result['similarity'] for result in self.training_results]
        rewards = [result['reward'] for result in self.training_results]
        positive_rewards = sum(1 for r in rewards if r > 0)
        
        return {
            'count': len(self.training_results),
            'avg_similarity': sum(similarities) / len(similarities),
            'avg_reward': sum(rewards) / len(rewards),
            'reward_rate': positive_rewards / len(rewards),
            'last_timestamp': self.training_results[-1]['timestamp'] if self.training_results else None
        }
    
    def run_training_session(self, prompts, manual_review=True):
        """
        Run a complete training session on a list of prompts.
        
        Args:
            prompts: List of prompts to use for training
            manual_review: Whether to pause for manual review of each prediction
            
        Returns:
            session_results: Results from the training session
        """
        session_results = []
        
        for prompt in prompts:
            # First phase: Make prediction
            prediction = self.make_prediction(prompt)
            
            if manual_review:
                print("\n" + "="*80)
                print(f"PROMPT: {prompt}")
                print("-"*80)
                print(f"PREDICTION: {prediction}")
                print("-"*80)
                input("Press Enter to continue and generate actual output...")
            
            # Second phase: Generate actual output
            input_bytes = prompt.encode('utf-8')
            input_tensor = torch.tensor([[byte for byte in input_bytes]], dtype=torch.long)
            
            output, _, _ = self.model(input_tensor)
            
            # Convert output tensor to text
            if isinstance(output, torch.Tensor):
                output = output.squeeze().detach().cpu().numpy()
                output_bytes = bytes([min(max(int(t), 0), 255) for t in output if t >= 0])
                actual_output = output_bytes.decode('utf-8', errors='replace')
            else:
                actual_output = str(output)
            
            # Third phase: Evaluate and apply reward
            reward, similarity = self.evaluate_output(prompt, actual_output)
            
            if manual_review:
                print(f"ACTUAL OUTPUT: {actual_output}")
                print("-"*80)
                print(f"Similarity: {similarity:.4f}, Reward: {reward:.4f}")
                
                # Allow user to override the reward
                user_input = input("Enter custom reward value (or press Enter to use calculated reward): ")
                if user_input.strip():
                    try:
                        reward = float(user_input)
                        print(f"Using custom reward: {reward}")
                        
                        # Update the last training result with custom reward
                        self.training_results[-1]['reward'] = reward
                        self.training_results[-1]['manual_override'] = True
                    except ValueError:
                        print("Invalid input, using calculated reward")
                
                input("Press Enter to apply reward and continue...")
            
            # Apply the reward
            self.apply_reward(reward)
            
            # Add to session results
            session_results.append({
                'prompt': prompt,
                'prediction': prediction,
                'actual_output': actual_output,
                'similarity': similarity,
                'reward': reward
            })
        
        # Save results
        self.save_training_results()
        
        print("\n" + "="*80)
        print("Training session completed!")
        stats = self.get_training_stats()
        print(f"Total training examples: {stats['count']}")
        print(f"Average similarity: {stats['avg_similarity']:.4f}")
        print(f"Average reward: {stats['avg_reward']:.4f}")
        print(f"Reward rate: {stats['reward_rate']:.4f}")
        
        return session_results


def run_introspection_training(model, sample_prompts=None, num_samples=610, manual_review=True, use_stable_softmax=False):
    """
    Run introspection training on a set of sample prompts.
    
    Args:
        model: The COCONUT model to train
        sample_prompts: Optional list of sample prompts to use (if None, uses default samples)
        num_samples: Number of samples to use if using default samples
        manual_review: Whether to pause for manual review of each prediction
        use_stable_softmax: Whether to use StableCrossEntropyLoss instead of standard CrossEntropyLoss
    
    Returns:
        trainer: The IntrospectionRewardTraining instance after training
    """
    # If stable softmax is requested, replace any CrossEntropyLoss with StableCrossEntropyLoss
    if use_stable_softmax and hasattr(model, 'criterion'):
        reduction = 'mean'
        if isinstance(model.criterion, CrossEntropyLoss):
            reduction = model.criterion.reduction
        model.criterion = StableCrossEntropyLoss(reduction=reduction)
        print("Replaced standard CrossEntropyLoss with StableCrossEntropyLoss for numerical stability")
    elif use_stable_softmax and hasattr(model, 'loss_fn'):
        reduction = 'mean'
        if isinstance(model.loss_fn, CrossEntropyLoss):
            reduction = model.loss_fn.reduction
        model.loss_fn = StableCrossEntropyLoss(reduction=reduction)
        print("Replaced standard CrossEntropyLoss with StableCrossEntropyLoss for numerical stability")
    # Initialize the trainer
    trainer = IntrospectionRewardTraining(model)
    
    # Default sample prompts if none provided
    if sample_prompts is None:
        sample_prompts = [
            "Tell me about artificial intelligence.",
            "Explain the concept of self-reflection in AI.",
            "What is the capital of France?",
            "Write a short poem about nature.",
            "Solve this math problem: 5 + 7 * 3",
            "Describe the process of photosynthesis.",
            "What are the ethical concerns of autonomous vehicles?",
            "Translate 'hello' to Spanish.",
            "Who was the first person to walk on the moon?",
            "How does a computer processor work?",
            "Create a simple recipe for chocolate cake.",
            "Explain quantum computing in simple terms.",
            "List the primary colors.",
            "What is the difference between weather and climate?",
            "Why do leaves change color in the fall?",
            "What is the Pythagorean theorem?",
            "Who wrote 'Pride and Prejudice'?",
            "What is the theory of relativity?",
            "Describe the water cycle.",
            "What causes a rainbow to form?",
    "You’re a chef preparing for a grand feast when the power goes out. How do you proceed with the meal?",
    "You’re hiking in the mountains and discover a hidden cave. What do you do next?",
    "You’re a mayor facing a sudden flood in your town. How do you respond to protect the residents?",
    "You’re stranded on a deserted island with a single tool of your choosing. What do you pick and how do you use it?",
    "You’re a writer who wakes up inside your own story. How do you navigate the plot?",
    "You’re captaining a sinking ship with a crew and passengers aboard. What’s your plan to save them?",
    "You’re a scientist who discovers a new species. How do you approach studying it?",
    "You’re a parent whose child claims to see a ghost. How do you handle the situation?",
    "You’re a musician invited to perform for a mysterious audience. How do you prepare your set?",
    "You’re a firefighter arriving at a burning building with cries from multiple floors. How do you prioritize your rescue?",
    "You’re a time traveler stuck in an unfamiliar era. How do you adapt to survive?",
    "You’re a farmer facing a drought with limited resources. How do you sustain your land?",
    "You’re a journalist uncovering a conspiracy. How do you decide to share your findings?",
    "You’re a pilot whose plane loses communication mid-flight. How do you guide it to safety?",
    "You’re a teacher whose students discover a hidden room in the school. How do you explore it with them?",
    "You’re a doctor in a remote village with a sudden outbreak. How do you manage the crisis?",
    "You’re a park ranger who finds an injured animal. How do you assist it?",
    "You’re a spy infiltrating a high-stakes gala. How do you gather the intel you need?",
    "You’re a photographer capturing a once-in-a-lifetime event. How do you frame your shots?",
    "You’re a baker tasked with feeding a festival crowd. How do you organize your baking?",
    "You’re a soldier lost behind enemy lines. How do you find your way back?",
    "You’re a designer creating for an eccentric client. How do you meet their wild demands?",
    "You’re a spaceship captain encountering an alien signal. How do you respond to it?",
    "You’re a librarian who finds a book that predicts the future. How do you handle its secrets?",
    "You’re a climber caught in an avalanche. How do you fight to survive?",
    "You’re a painter commissioned for a royal portrait. How do you approach the masterpiece?",
    "You’re a politician facing a public scandal. How do you address the crowd?",
    "You’re a zookeeper when an animal escapes its enclosure. How do you recapture it safely?",
    "You’re a sailor adrift in a storm-tossed sea. How do you regain control of your vessel?",
    "You’re a detective handed a case with no leads. How do you start your investigation?",
    "You’re a gardener discovering a rare plant in your backyard. How do you care for it?",
    "You’re a student who finds a locked box in the library. How do you uncover its contents?",
    "You’re a chef competing in a mystery ingredient contest. How do you craft your dish?",
    "You’re a traveler arriving in a city where no one speaks your language. How do you communicate?",
    "You’re a mechanic fixing a vehicle in the middle of nowhere. How do you improvise with limited tools?",
    "You’re a writer tasked with ending a centuries-old tale. How do you conclude the story?",
    "You’re a coach leading a team on the brink of defeat. How do you turn the game around?",
    "You’re a scientist stranded in a lab during a blizzard. How do you continue your research?",
    "You’re a parent planning a birthday party during a blackout. How do you keep the kids entertained?",
    "You’re a pilot flying through uncharted skies. How do you chart your course?",
    "You’re a doctor treating a patient with an unknown illness. How do you attempt to diagnose them?",
    "You’re a filmmaker shooting in a chaotic location. How do you capture the perfect scene?",
    "You’re a farmer whose crops are invaded by pests. How do you protect your harvest?",
    "You’re a musician performing with a broken instrument. How do you finish the show?",
    "You’re a teacher whose classroom is flooded. How do you adapt your lesson?",
    "You’re a hiker who finds an ancient artifact. How do you decide what to do with it?",
    "You’re a chef running out of ingredients mid-service. How do you satisfy your customers?",
    "You’re a sailor navigating through a fog so thick you can’t see the bow. How do you steer?",
    "You’re a painter whose studio catches fire. How do you save your work?",
    "You’re a writer whose characters start rewriting your story. How do you regain control?",
    "You’re a soldier tasked with defending a village under siege. How do you protect the people?",
    "You’re a doctor in a war zone with limited supplies. How do you treat the wounded?",
    "You’re a gardener whose plants begin to glow at night. How do you investigate the phenomenon?",
    "You’re a pilot whose co-pilot falls ill mid-flight. How do you manage the cockpit?",
    "You’re a chef whose kitchen is invaded by critics. How do you impress them?",
    "You’re a teacher whose students stage a rebellion. How do you restore order?",
    "You’re a climber facing a crumbling cliffside. How do you reach the top?",
    "You’re a musician hired for a secret gig. How do you tailor your performance?",
    "You’re a farmer whose livestock starts acting strangely. How do you address the mystery?",
    "You’re a spy whose cover is blown. How do you escape the enemy?",
    "You’re a writer invited to a debate with your own characters. How do you argue your case?",
    "You’re a sailor spotting a distant island. How do you decide whether to explore it?",
    "You’re a painter whose colors start mixing on their own. How do you finish the piece?",
    "You’re a doctor whose patient claims to be immortal. How do you test their story?",
    "You’re a coach whose team loses faith in you. How do you rally them?",
    "You’re a hiker stranded in a snowstorm. How do you find shelter?",
    "You’re a chef tasked with cooking for a rival. How do you approach the meal?",
    "You’re a teacher whose lesson is interrupted by a strange noise. How do you investigate?",
    "You’re a soldier discovering an enemy’s secret weapon. How do you report it?",
    "You’re a gardener whose flowers begin singing. How do you respond to the magic?",
    "You’re a pilot flying over a city that shouldn’t exist. How do you document it?",
    "You’re a writer whose pen starts writing on its own. How do you harness its power?",
    "You’re a musician whose audience vanishes mid-song. How do you finish the performance?",
    "You’re a farmer whose crops grow overnight. How do you manage the abundance?",
    "You’re a spy trapped in a locked room. How do you break free?",
    "You’re a painter whose subject comes to life. How do you interact with them?",
    "You’re a doctor treating a patient who refuses help. How do you convince them?",
    "You’re a sailor whose compass spins wildly. How do you find your way?",
    "You’re a coach facing a sudden rule change mid-game. How do you adapt your strategy?",
    "You’re a hiker who stumbles into a hidden village. How do you approach the locals?",
    "You’re a chef whose ingredients turn out to be enchanted. How do you cook with them?",
    "You’re a teacher whose students find a time capsule. How do you explore its contents?",
    "You’re a soldier tasked with delivering a message across a battlefield. How do you succeed?",
    "You’re a gardener whose soil starts shifting inexplicably. How do you stabilize it?",
    "You’re a pilot whose plane is boarded by stowaways. How do you handle them?",
    "You’re a writer whose story predicts real events. How do you use this power?",
    "You’re a musician whose instrument plays unfamiliar tunes. How do you master it?",
    "You’re a farmer whose barn is struck by lightning. How do you save your animals?",
    "You’re a spy who intercepts a coded message. How do you decipher it?",
    "You’re a painter whose canvas reveals hidden images. How do you interpret them?",
    "You’re a doctor in a town where everyone’s dreams are the same. How do you investigate?",
    "You’re a sailor whose crew mutinies. How do you regain their loyalty?",
    "You’re a coach whose star player disappears before the game. How do you adjust?",
    "You’re a hiker who finds a portal in the woods. How do you decide what to do with it?",
    "You’re a chef whose dish comes alive. How do you serve it?",
    "You’re a teacher whose classroom is visited by a historical figure. How do you engage them?",
    "You’re a soldier whose weapon malfunctions mid-battle. How do you improvise?",
    "You’re a gardener whose plants predict the weather. How do you use their forecasts?",
    "You’re a pilot flying through a sudden meteor shower. How do you navigate the chaos?",
    "You’re a writer whose manuscript is stolen. How do you recover it?",
    "You’re a musician whose song summons a storm. How do you control its effects?",
    "You’re a farmer whose fields are visited by strange lights. How do you investigate?",
    "You’re a spy who discovers a traitor in your ranks. How do you expose them?",
    "You’re a painter whose brush paints the future. How do you wield its power?",
    "You’re a doctor whose patient vanishes from their room and the window is open. What do you do next?", 
    "You’re a sailor who spots a ghost ship. How do you approach it?",
    "You’re a coach whose team faces a supernatural opponent. How do you prepare them?",
    "You’re a hiker whose map leads to treasure. How do you pursue it?",
    "You’re a chef whose kitchen is haunted. How do you cook amidst the chaos?",
    "You’re a teacher whose students gain superpowers overnight. How do you guide them?",
    "You’re a soldier tasked with guarding a mystical artifact. How do you protect it?",
    "You’re a gardener with roses that bloom with messages. How do you decode them?",
    "You’re a pilot whose plane enters a time warp. How do you return to your era?",
    "You’re a writer whose words alter reality. How do you rewrite the world?",
    "You’re a musician whose melody opens a portal. How do you explore what’s beyond?",
    "You’re a farmer whose animals start speaking. How do you communicate with them?",
    "You’re a spy whose gadgets begin malfunctioning. How do you complete the mission?",
    "You’re a painter whose colors summon spirits. How do you manage their presence?",
    "You’re a doctor whose clinic is overrun by mythical creatures. How do you treat them?",
    "You’re a sailor whose ship is pulled into a whirlpool. How do you escape?",
    "You’re a coach whose game is interrupted by time travelers. How do you handle the disruption?",
    "You’re a hiker who discovers a fountain of youth. How do you decide its fate?",
    "You’re a chef whose spices grant wishes. How do you season your next dish?",
    "You’re a teacher whose blackboard shows the future. How do you teach with it?",
    "You’re a soldier whose enemy offers peace unexpectedly. How do you respond?",
    "You’re a gardener whose trees bear golden fruit. How do you harvest them?",
    "You’re a pilot whose passengers vanish mid-flight. How do you solve the mystery?",
    "You’re a writer whose typewriter traps souls. How do you free them?",
    "You’re a musician whose audience turns into animals. How do you finish the concert?",
    "You’re a farmer whose land floats into the sky. How do you tend it?",
    "You’re a spy whose target disappears. How do you track them?",
    "You’re a painter whose portraits age instead of their subjects. How do you handle the curse?",
    "You’re a doctor whose medicine brings back memories. How do you prescribe it?",
    "You’re a sailor whose sea turns to glass. How do you navigate it?",
    "You’re a coach whose playbook predicts outcomes. How do you use it?",
    "You’re a hiker whose trail loops endlessly. How would you try to break the cycle?",
    "You’re a chef whose oven bakes clay. What clay sculpture would you create first?",
    "You’re a teacher whose students swap bodies. How do you maintain class?",
    "You’re a soldier whose shadow fights beside you. How do you command it?",
    "You’re a gardener whose soil whispers secrets. How do you listen?",
    "You’re a pilot flying a plane with a broken engine. How do you fly straight?",
    "You’re a writer whose ink binds fates. How do you pen the next chapter?",
    "You’re a musician whose flute calms animals. Would you use this ability for personal gain or use it to help others?",
    "You’re a farmer whose harvest feeds the monsters nearby. How do you prepare the feast for a potentially dangerous creature?", 
    "You’re a spy whose reflection ( a magical entity anchored to your body like a dual personality) betrays you. How do you approach your missions?",
    "You’re a painter whose frame traps viewers. How do you release them?",
    "You’re a doctor whose stethoscope hears thoughts. How do you diagnose the patient?",
    "You’re a sailor who is working on a boat while the magical waves sing lullabies. How do you stay awake?",
    "You’re a coach whose whistle summons legends. How do you lead them?",
    "You’re a hiker whose compass points to other's dreams. Do you chose to follow it?",
    "You’re a chef whose broth reveals truths. How do you serve it?",
    "You’re a teacher whose chalk draws portals. How do you explore them?",
    "You’re a soldier whose armor reflects the past. How do you wear it?",
    "You’re a gardener whose vines reach the moon. How do you climb them?",
    "You’re a pilot who sees clouds that form familiar faces. Who would they be?",
    "You’re a writer whose pages rewrite history. How do you edit them?",
    "You’re a musician whose notes paint the air. How do you compose?",
    "You’re a farmer whose windmill spins fate. How do you harness it?",
    "You’re a spy whose shadow moves alone. How do you control it?",
    "You’re a painter whose brush bends reality. How do you stroke it?",
    "You’re a doctor whose scalpel cuts time in half. How do you operate?",
    "You’re a sailor whose anchor pulls up islands. How do you choose one?",
    "You’re a coach whose team transcends dimensions. How do you train them?",
    "You’re a hiker whose echo builds bridges. How do you cross them?",
    "You’re a chef whose flames dance with spirits. How do you cook?",
    "You’re a teacher whose clock ticks backward. How do you time lessons?",
    "You’re a soldier whose bullets can change fate of all on Earth. How do you fire them?",
    "You’re a gardener whose water grows plants that contain other's memories. How do you water the plants?",
    "You’re a pilot whose plane's wings touch the stars. Where do you soar?",
    "You’re a writer whose margins hold worlds. How do you fill them?",
    "You’re a musician whose silence deafens gods. Which god would you want to drown out?",
    "You’re a farmer whose plow unearths time. Where do you till?",
    "You’re a spy whose whispers topple empires. Where do you speak?",
    "You’re a painter whose shadows cast light. How do you shade?",
    "You’re a doctor whose breath heals wounds. Who would you heal?",
    "You’re a sailor and there is a great fog that blocks your sight. How do you sail?",
    "You’re a coach whose cheers raise the dead. Where do you shout?",
    "You’re a hiker whose steps carve rivers. Where do you walk?",
    "You’re a chef whose salt seasons souls. What food would you sprinkle?",
    "You’re a teacher whose words shape futures. How do you lecture?",
    "You’re a soldier whose blood waters peace. How do you fight?",
    "You’re a gardener whose thorns guard secrets. How do you prune?",
    "You’re a pilot in frequent storms. How do you steer?",
    "You’re a writer whose commas pause wars. How do you punctuate?",
    "You’re a musician whose chords bind fates. How do you strum?",
    "You’re a farmer whose seeds sprout dreams. How do you sow?",
    "You’re a spy whose footsteps erase tracks. How do you sneak?",
    "You’re a painter whose frames hold time. How do you hang them?",
    "You’re a doctor whose touch rewinds clocks. How do you heal?",
    "You’re a sailor whose rudder steers the past. How do you turn?",
    "You’re a coach whose plays defy gravity. How do you call them?",
    "You’re a hiker whose shadow climbs alone. How do you follow?",
    "You’re a chef whose steam breathes life. How do you simmer?",
    "You’re a teacher whose . How do you anchor?",
    "You’re a soldier whose shield mirrors souls. How do you defend?",
    "You’re a gardener whose petals whisper names. How do you bloom?",
    "You’re a pilot whose altitude bends worlds. How do you rise?",
    "You’re a writer whose endings birth beginnings. How do you close?",
    "You’re a musician who is ready for a performance. How do you sing?",
    "You’re a farmer who reaps corn. How would you start the process?",
    "You’re a chef with only enough ingredients to make either a hearty soup or a decadent dessert for a group of hungry guests. Which dish do you choose to prepare?",
    "You’re hiking with a friend when you spot a bear. You can either climb a tree or play dead to avoid danger. Which action do you take?",
    "You’re a mayor of a small town facing a budget crisis. You can either cut funding for schools or the local hospital. Which do you choose?",
    "You’re stranded on a desert island with a radio that can either call for rescue or play music to keep your spirits up. Which function do you use?",
    "You’re a writer facing a deadline. You can either submit a polished short story or an unedited novel draft. Which do you send?",
    "You’re in charge of a sinking ship and can save either the passengers or the valuable cargo. Which do you prioritize?",
    "You’re a scientist with funding to research either a cure for a rare disease or a solution to climate change. Which project do you pursue?",
    "You’re a parent whose child is struggling. You can either hire a tutor or spend more time helping them yourself. Which do you choose?",
    "You’re a musician who can perform either an emotional ballad or an upbeat anthem at a concert. Which song do you pick?",
    "You’re a firefighter with time to save either a family’s pet or their photo albums from a burning house. Which do you rescue?",
    "You’re a time traveler who can either warn someone about a tragedy or explore a historical event. Which do you do?",
    "You’re a farmer with a limited water supply. You can either irrigate your crops or save it for your livestock. Which do you choose?",
    "You’re a journalist who can either expose a politician’s scandal or protect their privacy for a greater story later. Which path do you take?",
    "You’re a pilot with a malfunctioning plane. You can either attempt an emergency landing or parachute out with the crew. Which do you decide?",
    "You’re a teacher with one prize to give. Do you award it to the hardworking student or the naturally gifted one?",
    "You’re a doctor with one ventilator left. Do you give it to the elderly patient or the young one?",
    "You’re a park ranger who spots a fire. Do you try to extinguish it yourself or call for backup and risk it spreading?",
    "You’re a spy who can either steal enemy plans or rescue a captured ally. Which mission do you undertake?",
    "You’re a photographer who can either capture a rare animal or a stunning sunset. Which shot do you take?",
    "You’re a baker with time to make either bread for the homeless or pastries for a paying customer. Which do you bake?",
    "You’re a soldier who can either defend a strategic outpost or retreat to save your squad. Which do you choose?",
    "You’re a designer who can create either a practical gadget or a luxurious fashion piece. Which do you design?",
    "You’re a captain of a spaceship with failing oxygen. Do you ration it equally or prioritize the crew’s strongest members?",
    "You’re a librarian with one book to preserve in a flood. Do you save a rare manuscript or a children’s favorite?",
    "You’re a climber stuck in a storm. Do you push for the summit or descend to safety?",
    "You’re a painter with one canvas left. Do you paint a self-portrait or a landscape?",
    "You’re a politician who can either fund a new park or repair old roads. Which project do you back?",
    "You’re a zookeeper who can either feed the lions or clean the penguin habitat. Which task do you do?",
    "You’re a sailor lost at sea. Do you conserve your food or use it to bait fish?",
    "You’re a detective with two leads. Do you follow the suspicious stranger or the cryptic note?",
    "You’re a gardener with limited sunlight. Do you grow vegetables or flowers?",
    "You’re a student with one hour to study. Do you focus on math or history?",
    "You’re a chef with a single fresh fish. Do you grill it for a critic or share it with your team?",
    "You’re a traveler with one ticket. Do you visit a bustling city or a quiet village?",
    "You’re a mechanic with one spare part. Do you fix a customer’s car or your own?",
    "You’re a writer with one pen. Do you draft a poem or a letter to a friend?",
    "You’re a coach who can train either the star player or the struggling rookie. Which do you help?",
    "You’re a scientist with one test subject. Do you experiment on a plant or an animal?",
    "You’re a parent with one bedtime story. Do you tell a fairy tale or a real-life adventure?",
    "You’re a pilot with one fuel tank. Do you fly to a nearby island or risk a longer route home?",
    "You’re a doctor with one dose of medicine. Do you treat a friend or a stranger?",
    "You’re a filmmaker with one scene left to shoot. Do you capture a dramatic climax or a quiet moment?",
    "You’re a farmer with one seed. Do you plant it now or save it for next season?",
    "You’re a musician with one string left. Do you play a solo or repair your instrument?",
    "You’re a teacher with one lesson. Do you teach creativity or discipline?",
    "You’re a hiker with one match. Do you light a fire or save it for later?",
    "You’re a chef with one egg. Do you make an omelet or bake a cake?",
    "You’re a sailor with one rope. Do you tie down the sail or secure the anchor?",
    "You’re a painter with one color. Do you use red for passion or blue for calm?",
    "You’re a writer with one page. Do you tell a tragedy or a comedy?",
    "You’re a soldier with one bullet. Do you take a risky shot or hold your position?",
    "You’re a doctor with one bandage. Do you treat a small cut or save it for a worse injury?",
    "You’re a gardener with one rose. Do you keep it or give it away?",
    "You’re a pilot with one radio call. Do you contact help or send a distress signal?",
    "You’re a chef with one spice. Do you season with salt or chili?",
    "You’re a teacher with one marker. Do you write a formula or draw a picture?",
    "You’re a climber with one piton. Do you secure it now or save it for a harder section?",
    "You’re a musician with one note left. Do you end loud or soft?",
    "You’re a farmer with one chicken. Do you raise it for eggs or slaughter it for meat?",
    "You’re a spy with one gadget. Do you use a camera or a lockpick?",
    "You’re a writer with one word left. Do you write 'hope' or 'fear'?",
    "You’re a sailor with one flare. Do you signal now or wait for night?",
    "You’re a painter with one brush. Do you paint broad strokes or fine details?",
    "You’re a doctor with one syringe. Do you vaccinate or sedate?",
    "You’re a coach with one timeout. Do you call it now or save it for later?",
    "You’re a hiker with one map. Do you follow the river or the mountain path?",
    "You’re a chef with one potato. Do you mash it or fry it?",
    "You’re a teacher with one student. Do you encourage or challenge them?",
    "You’re a soldier with one grenade. Do you attack or defend?",
    "You’re a gardener with one watering can. Do you water the herbs or the trees?",
    "You’re a pilot with one parachute. Do you take it or give it to a passenger?",
    "You’re a writer with one chapter left. Do you resolve the plot or leave it open?",
    "You’re a musician with one drum. Do you play a rhythm or a single beat?",
    "You’re a farmer with one cow. Do you milk it or sell it?",
    "You’re a spy with one disguise. Do you impersonate a guard or a diplomat?",
    "You’re a painter with one frame. Do you display your work or gift it?",
    "You’re a doctor with one stethoscope. Do you check the heart or the lungs?",
    "You’re a sailor with one paddle. Do you row left or right?",
    "You’re a coach with one play. Do you go for a safe move or a bold one?",
    "You’re a hiker with one compass. Do you head north or south?",
    "You’re a chef with one tomato. Do you make sauce or a salad?",
    "You’re a teacher with one book. Do you read aloud or assign it?",
    "You’re a soldier with one flare gun. Do you signal allies or distract enemies?",
    "You’re a gardener with one shovel. Do you dig a new bed or turn the soil?",
    "You’re a pilot with one engine left. Do you fly high or low?",
    "You’re a writer with one deadline. Do you rush the ending or polish the start?",
    "You’re a musician with one microphone. Do you sing or speak?",
    "You’re a farmer with one acre. Do you plant wheat or corn?",
    "You’re a spy with one code. Do you encrypt a message or decode one?",
    "You’re a painter with one easel. Do you paint indoors or outside?",
    "You’re a doctor with one scalpel. Do you operate now or wait?",
    "You’re a sailor with one sail. Do you raise it or repair it?",
    "You’re a coach with one substitution. Do you swap offense or defense?",
    "You’re a hiker with one rope. Do you climb up or rappel down?",
    "You’re a chef with one onion. Do you chop it or caramelize it?",
    "You’re a teacher with one quiz. Do you test knowledge or creativity?",
    "You’re a soldier with one map. Do you advance or retreat?",
    "You’re a gardener with one seed packet. Do you plant carrots or beans?",
    "You’re a pilot with one landing gear. Do you land now or circle?",
    "You’re a writer with one twist. Do you surprise the hero or the villain?",
    "You’re a musician with one chord. Do you play it major or minor?",
    "You’re a farmer with one fence. Do you protect the crops or the animals?",
    "You’re a spy with one bullet. Do you shoot to kill or to wound?",
    "You’re a painter with one light. Do you work at dawn or dusk?",
    "You’re a doctor with one pill. Do you ease pain or cure infection?",
    "You’re a sailor with one net. Do you fish or mend it?",
    "You’re a coach with one goal. Do you aim to win or to teach?",
    "You’re a hiker with one shoe. Do you walk barefoot or wait?",
    "You’re a chef with one burner. Do you cook rice or meat?",
    "You’re a teacher with one rule. Do you enforce silence or participation?",
    "You’re a soldier with one radio. Do you call for help or report in?",
    "You’re a gardener with one pot. Do you grow basil or mint?",
    "You’re a pilot with one wing. Do you glide or eject?",
    "You’re a writer with one reader. Do you inspire or entertain them?",
    "You’re a musician with one audience. Do you perform for joy or sorrow?",
    "You’re a farmer with one barn. Do you store hay or shelter animals?",
    "You’re a spy with one contact. Do you trust them or go solo?",
    "You’re a painter with one wall. Do you mural or leave it blank?",
    "You’re a doctor with one bed. Do you admit a child or an elder?",
    "You’re a sailor with one compass. Do you sail east or west?",
    "You’re a coach with one speech. Do you motivate or strategize?",
    "You’re a hiker with one trail. Do you take the steep or the scenic?",
    "You’re a chef with one plate. Do you serve hot or cold?",
    "You’re a teacher with one desk. Do you sit or let a student use it?",
    "You’re a soldier with one shield. Do you protect yourself or a comrade?",
    "You’re a gardener with one trellis. Do you grow vines or roses?",
    "You’re a pilot with one passenger. Do you calm them or focus on flying?",
    "You’re a writer with one pen stroke. Do you start or finish?",
    "You’re a musician with one melody. Do you repeat it or vary it?",
    "You’re a farmer with one plow. Do you till now or wait for rain?",
    "You’re a spy with one exit. Do you sneak out or fight through?",
    "You’re a painter with one shadow. Do you highlight or darken it?",
    "You’re a doctor with one mask. Do you wear it or give it away?",
    "You’re a sailor with one wave. Do you ride it or brace for it?",
    "You’re a coach with one whistle. Do you start or stop the play?",
    "You’re a hiker with one ridge. Do you cross it or camp below?",
    "You’re a chef with one knife. Do you slice or dice?",
    "You’re a teacher with one pointer. Do you point up or down?",
    "You’re a soldier with one order. Do you obey or question it?",
    "You’re a gardener with one sprout. Do you nurture it or replant?",
    "You’re a pilot with one cloud. Do you fly through or around it?",
    "You’re a writer with one title. Do you make it bold or subtle?",
    "You’re a musician with one beat. Do you speed it up or slow it down?",
    "You’re a farmer with one well. Do you drink or irrigate?",
    "You’re a spy with one clue. Do you follow it or ignore it?",
    "You’re a painter with one stroke. Do you go straight or curve?",
    "You’re a doctor with one glove. Do you use it or save it?",
    "You’re a sailor with one wind. Do you harness it or wait?",
    "You’re a coach with one chance. Do you take a risk or play safe?",
    "You’re a hiker with one view. Do you stop to admire or keep moving?",
    "You’re a chef with one herb. Do you use parsley or thyme?",
    "You’re a teacher with one question. Do you ask it or answer it?",
    "You’re a soldier with one flag. Do you raise it or guard it?",
    "You’re a gardener with one bloom. Do you pick it or let it grow?",
    "You’re a pilot with one star. Do you navigate by it or ignore it?",
    "You’re a writer with one scene. Do you set it day or night?",
    "You’re a musician with one lyric. Do you whisper or shout it?",
    "You’re a farmer with one gate. Do you open it or lock it?",
    "You’re a spy with one shadow. Do you hide in it or avoid it?",
    "You’re a painter with one hue. Do you blend it or leave it pure?",
    "You’re a doctor with one chart. Do you study it or trust your gut?",
    "You’re a sailor with one knot. Do you tie it or cut it?",
    "You’re a coach with one player. Do you push them or rest them?",
    "You’re a hiker with one peak. Do you summit or detour?",
    "You’re a chef with one broth. Do you sip it or season it?",
    "You’re a teacher with one lesson plan. Do you stick to it or improvise?",
    "You’re a soldier with one boot. Do you march or rest?",
    "You’re a gardener with one leaf. Do you press it or plant it?",
    "You’re a pilot with one gauge. Do you trust it or fly blind?",
    "You’re a writer with one character. Do you save them or doom them?",
    "You’re a musician with one tempo. Do you keep it or change it?",
    "You’re a farmer with one egg. Do you hatch it or cook it?",
    "You’re a spy with one phone. Do you call home or your handler?",
    "You’re a painter with one line. Do you cross it or follow it?",
    "You’re a doctor with one light. Do you shine it high or low?",
    "You’re a sailor with one oar. Do you paddle or drift?",
    "You’re a coach with one drill. Do you perfect it or move on?",
    "You’re a hiker with one fork. Do you go left or right?",
    "You’re a chef with one loaf. Do you slice it or toast it?",
    "You’re a teacher with one clock. Do you speed up or slow down?",
    "You’re a soldier with one trench. Do you dig deeper or abandon it?",
    "You’re a gardener with one berry. Do you eat it or plant it?",
    "You’re a pilot with one signal. Do you respond or ignore it?",
    "You’re a writer with one ending. Do you make it happy or sad?",
    "You’re a musician with one silence. Do you break it or let it linger?",
    "You’re a farmer with one lamb. Do you shear it or feed it?",
    "You’re a spy with one mask. Do you wear it or burn it?",
    "You’re a painter with one canvas size. Do you go big or small?",
    "You’re a doctor with one needle. Do you stitch or draw blood?",
    "You’re a sailor with one horizon. Do you chase it or turn back?",
    "You’re a coach with one bench. Do you sit or stand?",
    "You’re a hiker with one stream. Do you cross it or follow it?",
    "You’re a chef with one sauce. Do you drizzle it or dip?",
    "You’re a teacher with one board. Do you write or erase?",
    "You’re a soldier with one hill. Do you take it or bypass it?",
    "You’re a gardener with one vine. Do you prune it or let it climb?",
    "You’re a pilot with one storm. Do you fly over or under it?",
    "You’re a writer with one quote. Do you open with it or close?",
    "You’re a musician with one stage. Do you center or edge it?",
    "You’re a farmer with one horse. Do you ride it or plow with it?",
    "You’re a spy with one key. Do you unlock or hide it?",
    "You’re a painter with one model. Do you pose them or let them rest?",
    "You’re a doctor with one room. Do you treat or diagnose?",
    "You’re a sailor with one tide. Do you sail with it or against it?",
    "You’re a coach with one score. Do you defend it or push for more?",
    "You’re a hiker with one cave. Do you explore or avoid it?",
    "You’re a chef with one lemon. Do you squeeze it or zest it?",
    "You’re a teacher with one chair. Do you offer it or take it?",
    "You’re a soldier with one blade. Do you sharpen it or sheath it?",
    "You’re a gardener with one bulb. Do you plant it shallow or deep?",
    "You’re a pilot with one runway. Do you land short or long?",
    "You’re a writer with one mystery. Do you solve it or deepen it?",
    "You’re a musician with one harmony. Do you join it or solo?",
    "You’re a farmer with one silo. Do you fill it or empty it?",
    "You’re a spy with one mirror. Do you signal or check behind?",
    "You’re a painter with one palette. Do you mix or keep it pure?",
    "You’re a doctor with one cast. Do you set a leg or an arm?",
    "You’re a sailor with one anchor. Do you drop it or haul it?",
    "You’re a coach with one timeout left. Do you rally or reset?",
    "You’re a hiker with one ridgepole. Do you pitch a tent or keep hiking?",
    "You’re a chef with one carrot. Do you roast it or shred it?",
    "You’re a teacher with one projector. Do you show slides or a video?",
    "You’re a soldier with one outpost. Do you reinforce or abandon it?",
    "You’re a gardener with one sprinkler. Do you water now or later?",
    "You’re a pilot with one beacon. Do you follow it or chart your own course?",
    "You’re a writer with one villain. Do you redeem or defeat them?",
    "You’re a musician with one encore. Do you repeat or improvise?",
    "You’re a farmer with one calf. Do you raise it or trade it?",
    "You’re a spy with one cipher. Do you crack it or create one?",
    "You’re a painter with one sketch. Do you refine it or start anew?",
    "You’re a doctor with one thermometer. Do you check fever or chill?",
    "You’re a sailor with one sextant. Do you navigate or teach it?",
    "You’re a coach with one star. Do you feature them or bench them?",
    "You’re a hiker with one pass. Do you cross now or wait out weather?",
    "You’re a chef with one mushroom. Do you sauté it or stuff it?",
    "You’re a teacher with one map. Do you trace routes or mark cities?",
    "You’re a soldier with one patrol. Do you scout ahead or guard rear?",
    "You’re a gardener with one hedge. Do you trim it or let it grow wild?",
    "You’re a pilot with one flare. Do you signal day or night?",
    "You’re a writer with one flashback. Do you place it early or late?",
    "You’re a musician with one riff. Do you loop it or build on it?",
    "You’re a farmer with one pond. Do you fish it or irrigate from it?",
    "You’re a spy with one safehouse. Do you hide there or move on?",
    "You’re a painter with one texture. Do you smooth it or rough it?",
    "You’re a doctor with one X-ray. Do you scan head or chest?",
    "You’re a sailor with one buoy. Do you tie to it or pass it?",
    "You’re a coach with one playbook. Do you stick to it or adapt?",
    "You’re a hiker with one bridge. Do you cross or find another way?",
    "You’re a chef with one clove. Do you mince it or roast it?",
    "You’re a teacher with one globe. Do you spin it or point?",
    "You’re a soldier with one wall. Do you breach it or climb it?",
    "You’re a gardener with one fern. Do you pot it or plant it?",
    "You’re a pilot with one altimeter. Do you climb or descend?",
    "You’re a writer with one prophecy. Do you fulfill or defy it?",
    "You’re a musician with one echo. Do you amplify or mute it?",
    "You’re a farmer with one goat. Do you milk it or graze it?",
    "You’re a spy with one signal. Do you send it or wait?",
    "You’re a painter with one glaze. Do you apply it or scrape it?",
    "You’re a doctor with one brace. Do you support a knee or wrist?",
    "You’re a sailor with one current. Do you ride it or fight it?",
    "You’re a coach with one cheer. Do you shout it or whisper it?",
    "You’re a hiker with one slope. Do you ascend or skirt it?",
    "You’re a chef with one pepper. Do you grind it or slice it?",
    "You’re a teacher with one chalk. Do you write big or small?",
    "You’re a soldier with one flare pistol. Do you aim high or low?",
    "You’re a gardener with one twig. Do you stake it or burn it?",
    "You’re a pilot with one gust. Do you bank or hold steady?",
    "You’re a writer with one dialogue. Do you make it sharp or soft?",
    "You’re a musician with one trill. Do you extend or cut it?",
    "You’re a farmer with one trough. Do you fill it with water or feed?",
    "You’re a spy with one drop. Do you make it or skip it?",
    "You’re a painter with one drip. Do you catch it or let it fall?",
    "You’re a doctor with one swab. Do you clean or test?",
    "You’re a sailor with one gull. Do you follow it or ignore it?",
    "You’re a coach with one lap. Do you time it or watch form?",
    "You’re a hiker with one marker. Do you trust it or blaze your own?",
    "You’re a chef with one garlic. Do you crush it or peel it?",
    "You’re a teacher with one bell. Do you ring it or let it sit?",
    "You’re a soldier with one ridge. Do you hold it or flank it?",
    "You’re a gardener with one bud. Do you pinch it or let it bloom?",
    "You’re a pilot with one fog bank. Do you enter or avoid it?",
    "You’re a writer with one metaphor. Do you stretch it or drop it?",
    "You’re a musician with one pause. Do you hold it or fill it?",
    "You’re a farmer with one pig. Do you fatten it or breed it?",
    "You’re a spy with one lens. Do you zoom in or scan wide?",
    "You’re a painter with one wash. Do you layer it or thin it?",
    "You’re a doctor with one vial. Do you inject or analyze?",
    "You’re a sailor with one swell. Do you surf it or brace?",
    "You’re a coach with one huddle. Do you inspire or instruct?",
    "You’re a hiker with one cliff. Do you climb it or edge along?",
    "You’re a chef with one lime. Do you juice it or garnish with it?",
    "You’re a teacher with one ruler. Do you measure or point?",
    "You’re a soldier with one flare pistol. Do you aim high or low?",
    "You’re a gardener with one twig. Do you stake it or burn it?",
    "You’re a pilot with one gust. Do you bank or hold steady?",
    "You’re a writer with one dialogue. Do you make it sharp or soft?",
    "You’re a musician with one trill. Do you extend or cut it?",
    "You’re a farmer with one trough. Do you fill it with water or feed?",
    "You’re a spy with one drop. Do you make it or skip it?",
    "You’re a painter with one drip. Do you catch it or let it fall?",
    "You’re a doctor with one swab. Do you clean or test?",
    "You’re a sailor with one gull. Do you follow it or ignore it?",
    "You’re a coach with one lap. Do you time it or watch form?",
    "You’re a hiker with one marker. Do you trust it or blaze your own?",
    "You’re a chef with one garlic. Do you crush it or peel it?",
    "You’re a teacher with one bell. Do you ring it or let it sit?",
    "You’re a soldier with one ridge. Do you hold it or flank it?",
    "You’re a gardener with one bud. Do you pinch it or let it bloom?",
    "You’re a pilot with one fog bank. Do you enter or avoid it?",
    "You’re a writer with one metaphor. Do you stretch it or drop it?",
    "You’re a musician with one pause. Do you hold it or fill it?",
    "You’re a farmer with one pig. Do you fatten it or breed it?",
    "You’re a spy with one lens. Do you zoom in or scan wide?",
    "You’re a painter with one wash. Do you layer it or thin it?",
    "You’re a doctor with one vial. Do you inject or analyze?",
    "You’re a sailor with one swell. Do you surf it or brace?",
    "You’re a coach with one huddle. Do you inspire or instruct?",
    "You’re a hiker with one cliff. Do you climb it or edge along?",
    "You’re a chef with one lime. Do you juice it or garnish with it?",
    "You’re a teacher with one ruler. Do you measure or point?",
    "You’re a soldier with one mine. Do you detonate or disarm it?",
    "You’re a gardener with one weed. Do you pull it or spray it?",
    "You’re a pilot with one wind shear. Do you adjust or power through?",
    "You’re a writer with one cliffhanger. Do you resolve it or extend it?",
    "You’re a musician with one fade. Do you draw it out or cut it short?",
    "You’re a farmer with one duck. Do you keep it for eggs or feathers?",
    "You’re a spy with one wire. Do you tap it or cut it?",
    "You’re a painter with one smudge. Do you blend it or wipe it?",
    "You’re a doctor with one pulse. Do you check it or assume it?",
    "You’re a sailor with one squall. Do you reef sails or ride it out?",
    "You’re a coach with one foul. Do you argue it or accept it?",
    "You’re a hiker with one ravine. Do you jump it or detour?",
    "You’re a chef with one basil leaf. Do you tear it or leave it whole?",
    "You’re a teacher with one eraser. Do you clean the board or save it?",
    "You’re a soldier with one ambush. Do you spring it or wait?",
    "You’re a gardener with one fruit. Do you eat it or save the seeds?",
    "You’re a pilot with one turbulence patch. Do you climb above or below?",
    "You’re a writer with one secret. Do you reveal it or hint at it?",
    "You’re a musician with one crescendo. Do you peak early or late?",
    "You’re a farmer with one beehive. Do you harvest honey or let it grow?",
    "You’re a spy with one dossier. Do you read it or burn it?",
    "You’re a painter with one crack. Do you repair it or feature it?",
    "You’re a doctor with one splint. Do you apply it or instruct?",
    "You’re a sailor with one leak. Do you patch it or bail water?",
    "You’re a coach with one injury. Do you submit or push through?",
    "You’re a hiker with one boulder. Do you climb over or go around?",
    "You’re a chef with one scallion. Do you chop it or grill it?",
    "You’re a teacher with one pen. Do you lend it or keep it?",
    "You’re a soldier with one smoke grenade. Do you cover retreat or advance?",
    "You’re a gardener with one thorn. Do you remove it or leave it?",
    "You’re a pilot with one lightning strike. Do you divert or press on?",
    "You’re a writer with one betrayal. Do you foreshadow or surprise?",
    "You’re a musician with one dissonance. Do you resolve or lean into it?",
    "You’re a farmer with one turkey. Do you raise it or roast it?",
    "You’re a spy with one footprint. Do you follow it or erase it?",
    "You’re a painter with one tear. Do you fix it or paint over?",
    "You’re a doctor with one cough. Do you medicate or monitor?",
    "You’re a sailor with one foghorn. Do you sound it or save it?",
    "You’re a coach with one tiebreaker. Do you go offense or defense?",
    "You’re a hiker with one shortcut. Do you risk it or stay on path?",
    "You’re a chef with one olive. Do you pit it or brine it?",
    "You’re a teacher with one window. Do you open it or shade it?",
    "You’re a soldier with one flare signal. Do you fire it or hold?",
    "You’re a gardener with one moss patch. Do you grow it or scrape it?",
    "You’re a pilot with one bird strike. Do you inspect or fly on?",
    "You’re a writer with one deadline extension. Do you take it or submit?",
    "You’re a musician with one broken string. Do you replace or adapt?",
    "You’re a farmer with one rabbit. Do you cage it or let it roam?",
    "You’re a spy with one decoy. Do you deploy it or save it?",
    "You’re a painter with one spill. Do you clean it or incorporate it?",
    "You’re a doctor with one rash. Do you treat it or observe?",
    "You’re a sailor with one driftwood. Do you grab it or leave it?",
    "You’re a coach with one overtime. Do you pace or push?",
    "You’re a hiker with one sunset. Do you camp or keep going?",
    "You’re a chef with one ginger root. Do you grate it or steep it?",
    "You’re a teacher with one test. Do you grade it or review it?",
    "You’re a soldier with one checkpoint. Do you hold or bypass?",
    "You’re a gardener with one slug. Do you remove it or ignore it?",
    "You’re a pilot with one ice patch. Do you de-ice or reroute?",
    "You’re a writer with one typo. Do you fix it or leave it?",
    "You’re a musician with one feedback loop. Do you stop or ride it?",
    "You’re a farmer with one crow. Do you scare it or feed it?",
    "You’re a spy with one blind spot. Do you cover it or exploit it?",
    "You’re a painter with one streak. Do you blend or accent it?",
    "You’re a doctor with one fever. Do you cool it or sweat it out?",
    "You’re a sailor with one shark. Do you evade or observe?",
    "You’re a coach with one penalty. Do you challenge or accept?",
    "You’re a hiker with one snake. Do you freeze or back away?",
    "You’re a chef with one chili. Do you seed it or leave it whole?",
    "You’re a teacher with one absentee. Do you wait or proceed?",
    "You’re a soldier with one ceasefire. Do you trust it or prepare?",
    "You’re a gardener with one aphid. Do you squash it or wash it?",
    "You’re a pilot with one hailstorm. Do you climb or descend?",
    "You’re a writer with one plot hole. Do you patch it or rewrite?",
    "You’re a musician with one flat note. Do you retune or play through?",
    "You’re a farmer with one fox. Do you trap it or guard against it?",
    "You’re a spy with one tail. Do you lose them or confront?",
    "You’re a painter with one bubble. Do you pop it or paint around?",
    "You’re a doctor with one limp. Do you brace it or rest it?",
    "You’re a sailor with one whale. Do you follow or steer clear?",
    "You’re a coach with one rookie. Do you start them or bench them?",
    "You’re a hiker with one storm cloud. Do you shelter or outrun it?",
    "You’re a chef with one leek. Do you soup it or sauté it?",
    "You’re a teacher with one latecomer. Do you reprimand or welcome?",
    "You’re a soldier with one deserter. Do you pursue or report?",
    "You’re a gardener with one frost. Do you cover plants or harvest?",
    "You’re a pilot with one crosswind. Do you adjust or hold course?",
    "You’re a writer with one critic. Do you heed them or ignore?",
    "You’re a musician with one clap. Do you bow or continue?",
    "You’re a farmer with one stray dog. Do you adopt or shoo it?",
    "You’re a spy with one double agent. Do you expose or use them?",
    "You’re a painter with one warp. Do you stretch or replace?",
    "You’re a doctor with one sneeze. Do you isolate or dismiss?",
    "You’re a sailor with one reef. Do you navigate or anchor?",
    "You’re a coach with one fan. Do you play to them or the team?",
    "You’re a hiker with one bear track. Do you follow or retreat?",
    "You’re a chef with one radish. Do you slice it or pickle it?",
    "You’re a teacher with one whisper. Do you quiet it or encourage?",
    "You’re a soldier with one prisoner. Do you interrogate or release?",
    "You’re a gardener with one ladybug. Do you keep it or move it?",
    "You’re a pilot with one glitch. Do you reboot or manual override?",
    "You’re a writer with one fan letter. Do you reply or archive?",
    "You’re a musician with one hiss. Do you adjust or ignore?",
    "You’re a farmer with one hawk. Do you watch it or scare it?",
    "You’re a spy with one leak. Do you plug it or misdirect?",
    "You’re a painter with one fade. Do you touch up or leave it?",
    "You’re a doctor with one bruise. Do you ice it or heat it?",
    "You’re a sailor with one starboard light. Do you fix it or sail on?"
    ]
        
        # Limit to specified number of samples
        if num_samples < len(sample_prompts):
            import random
            sample_prompts = random.sample(sample_prompts, num_samples)
    
    # Run the training session
    trainer.run_training_session(sample_prompts, manual_review=manual_review)
    
    return trainer


# Example of running a training session
if __name__ == "__main__" and "introspection_training" in sys.argv:
    import sys
    # Create a model instance (if running standalone)
    config = namedtuple("Config", [])()
    continuous_model = nn.Sequential(
        nn.Linear(256, 64),
        nn.ReLU()
    )
    
    # Create the CoconutBinaryLatentModel
    coconut_model = CoconutBinaryLatentModel(
        continuous_model=continuous_model,
        latent_transformer=CoconutBinaryLatentModel,
        local_encoder=CoconutBinaryLatentModel.multiencoder,
        input_dim=64,
        hidden_dim=32
    )
    
    # Determine whether to use stable softmax
    use_stable_softmax = "--use-stable-softmax" in sys.argv
    
    # Run introspection training
    trainer = run_introspection_training(
        coconut_model,
        use_stable_softmax=use_stable_softmax
    )
    
    # Save training results
    results_path = trainer.save_training_results()
    print(f"Training results saved to: {results_path}")
    

    save_checkpoint("introspection_finalStep")
    # Play sound to indicate completion
    play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
    print("Introspection training completed!")
    
