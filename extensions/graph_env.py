#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Environment for RL Training
Wraps semantic graph memory operations into an RL-compatible interface.
Integrates with PPO/GRPO trainers and logs trace buffers for replay.
"""

import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
import numpy as np
from pathlib import Path

# Import existing Memory-R1 components
try:
    from memory_r1_modular import (
        MemoryR1Enhanced, GraphBuilder, GraphMemoryBank, 
        GraphOperation, GraphTriple, GraphFragment
    )
    MEMORY_R1_AVAILABLE = True
except ImportError:
    MEMORY_R1_AVAILABLE = False
    print("⚠️ Memory-R1 modular system not available")

@dataclass
class EnvironmentState:
    """RL environment state representation"""
    graph_features: np.ndarray
    memory_stats: Dict[str, float]
    context_features: np.ndarray
    turn_id: int
    timestamp: datetime

@dataclass
class ActionResult:
    """Result of applying an action in the environment"""
    reward: float
    new_state: EnvironmentState
    done: bool
    info: Dict[str, Any]

class GraphMemoryEnv:
    """RL-compatible environment for semantic graph memory operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize Memory-R1 system
        if MEMORY_R1_AVAILABLE:
            self.memory_system = MemoryR1Enhanced(config.get("memory_r1", {}))
        else:
            # Fallback implementation
            self.graph = nx.MultiDiGraph()
            self.current_turn = 0
        
        # Environment configuration
        self.max_turns = config.get("max_turns", 100)
        self.observation_dim = config.get("observation_dim", 64)
        self.action_space_size = 4  # ADD_NODE, MERGE_EDGE, DELETE_SUBGRAPH, NOOP
        
        # Training dataset
        self.dataset = self._load_training_dataset(config.get("dataset_path", "training_data.json"))
        self.current_episode = 0
        
        # Trace buffer for replay
        self.trace_buffer = []
        self.episode_traces = []
        
        # Reward calculation components
        self.reward_weights = {
            "qa_accuracy": 0.4,
            "graph_integrity": 0.3,
            "memory_efficiency": 0.2,
            "operation_success": 0.1
        }
        
        print("🎮 Graph Memory Environment initialized")
        print(f"   Dataset size: {len(self.dataset)}")
        print(f"   Max turns per episode: {self.max_turns}")
        print(f"   Action space: {self.action_space_size}")
    
    def reset(self) -> EnvironmentState:
        """Reset environment for new episode"""
        
        if MEMORY_R1_AVAILABLE:
            # Reset Memory-R1 system
            self.memory_system = MemoryR1Enhanced(self.config.get("memory_r1", {}))
            self.current_turn = 0
        else:
            # Fallback reset
            self.graph.clear()
            self.current_turn = 0
        
        # Clear episode traces
        if self.episode_traces:
            self.trace_buffer.append({
                "episode": self.current_episode,
                "traces": self.episode_traces.copy(),
                "episode_reward": sum(trace.get("reward", 0) for trace in self.episode_traces),
                "episode_length": len(self.episode_traces)
            })
        
        self.episode_traces = []
        self.current_episode += 1
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done, info"""
        
        if self.current_turn >= len(self.dataset):
            return self._get_observation(), 0.0, True, {"error": "Episode complete"}
        
        try:
            # Get current input from dataset
            current_input = self.dataset[self.current_turn]
            input_text = current_input.get("text", "")
            expected_qa_accuracy = current_input.get("qa_accuracy", 0.5)
            
            # Decode action
            graph_operation = self._decode_action(action)
            
            # Apply operation using Memory-R1 system
            if MEMORY_R1_AVAILABLE:
                # Process input with Memory-R1
                result = self.memory_system.process_input(input_text, expected_qa_accuracy)
                
                # Calculate reward based on multiple factors
                reward = self._calculate_composite_reward(result, expected_qa_accuracy, graph_operation)
                
                # Get new observation
                new_state = self._get_observation()
                
                # Check if episode is done
                done = (self.current_turn >= self.max_turns - 1 or 
                       self.current_turn >= len(self.dataset) - 1)
                
                # Prepare info
                info = {
                    "turn": self.current_turn,
                    "operation": graph_operation.value,
                    "facts_extracted": len(result.get("extracted_facts", [])),
                    "graph_operations": result.get("graph_operations", []),
                    "qa_accuracy": expected_qa_accuracy,
                    "memory_efficiency": self._calculate_memory_efficiency(),
                    "graph_integrity": self._calculate_graph_integrity()
                }
                
                # Record trace
                trace_entry = {
                    "turn": self.current_turn,
                    "input": input_text,
                    "action": action,
                    "operation": graph_operation.value,
                    "reward": reward,
                    "result": result,
                    "info": info,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.episode_traces.append(trace_entry)
                
            else:
                # Fallback implementation
                reward = np.random.random() * 0.5  # Dummy reward
                new_state = self._get_observation()
                done = self.current_turn >= self.max_turns - 1
                info = {"turn": self.current_turn, "operation": graph_operation.value}
            
            self.current_turn += 1
            
            return new_state, reward, done, info
        
        except Exception as e:
            return self._get_observation(), -0.5, True, {"error": str(e)}
    
    def _get_observation(self) -> EnvironmentState:
        """Get current environment observation"""
        
        if MEMORY_R1_AVAILABLE:
            # Get graph features from Memory-R1 system
            graph_stats = self.memory_system.get_system_status()
            
            # Create graph feature vector
            graph_features = np.array([
                graph_stats["module_status"]["graph_memory"]["nodes"] / 100.0,  # Normalize
                graph_stats["module_status"]["graph_memory"]["edges"] / 100.0,
                graph_stats["module_status"]["graph_memory"]["fragments"] / 50.0,
                graph_stats["module_status"]["trace_buffer"]["utilization"],
                len(self.memory_system.graph_memory.entity_index) / 100.0,
                len(self.memory_system.graph_memory.relation_index) / 20.0
            ])
            
            # Pad or truncate to observation_dim
            if len(graph_features) < self.observation_dim:
                graph_features = np.pad(graph_features, (0, self.observation_dim - len(graph_features)))
            else:
                graph_features = graph_features[:self.observation_dim]
            
            # Memory statistics
            memory_stats = {
                "total_extractions": graph_stats["system_stats"]["total_extractions"],
                "total_operations": graph_stats["system_stats"]["total_operations"],
                "total_rewards": graph_stats["system_stats"]["total_rewards"],
                "current_turn": self.current_turn
            }
            
            # Context features (simplified)
            context_features = np.array([
                self.current_turn / self.max_turns,  # Progress
                len(self.episode_traces) / self.max_turns,  # Episode progress
                self._calculate_memory_efficiency(),
                self._calculate_graph_integrity()
            ])
            
        else:
            # Fallback observation
            graph_features = np.random.random(self.observation_dim)
            memory_stats = {"current_turn": self.current_turn}
            context_features = np.array([self.current_turn / self.max_turns, 0.5, 0.5, 0.5])
        
        return EnvironmentState(
            graph_features=graph_features,
            memory_stats=memory_stats,
            context_features=context_features,
            turn_id=self.current_turn,
            timestamp=datetime.now()
        )
    
    def _load_training_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load training dataset"""
        
        try:
            if Path(dataset_path).exists():
                with open(dataset_path, "r") as f:
                    return json.load(f)
            else:
                # Create sample dataset
                return [
                    {"text": "Paris is the capital of France.", "qa_accuracy": 0.9},
                    {"text": "France is located in Europe.", "qa_accuracy": 0.8},
                    {"text": "The Eiffel Tower is in Paris.", "qa_accuracy": 0.7},
                    {"text": "Europe contains many countries.", "qa_accuracy": 0.6},
                    {"text": "London is the capital of England.", "qa_accuracy": 0.9},
                    {"text": "England is part of the United Kingdom.", "qa_accuracy": 0.8},
                    {"text": "Shakespeare was born in England.", "qa_accuracy": 0.7},
                    {"text": "Python is a programming language.", "qa_accuracy": 0.8},
                    {"text": "Machine learning uses algorithms.", "qa_accuracy": 0.7},
                    {"text": "Neural networks are used in AI.", "qa_accuracy": 0.8}
                ]
        except Exception as e:
            print(f"⚠️ Failed to load dataset: {e}")
            return []
    
    def _decode_action(self, action: int) -> GraphOperation:
        """Decode integer action to graph operation"""
        
        action_mapping = {
            0: GraphOperation.ADD_NODE,
            1: GraphOperation.MERGE_EDGE,
            2: GraphOperation.DELETE_SUBGRAPH,
            3: GraphOperation.NOOP
        }
        
        return action_mapping.get(action, GraphOperation.NOOP)
    
    def _calculate_composite_reward(self, result: Dict[str, Any], expected_qa: float, 
                                  operation: GraphOperation) -> float:
        """Calculate composite reward for RL training"""
        
        # QA accuracy component
        qa_reward = expected_qa * self.reward_weights["qa_accuracy"]
        
        # Graph integrity component
        graph_integrity = self._calculate_graph_integrity()
        integrity_reward = graph_integrity * self.reward_weights["graph_integrity"]
        
        # Memory efficiency component
        memory_efficiency = self._calculate_memory_efficiency()
        efficiency_reward = memory_efficiency * self.reward_weights["memory_efficiency"]
        
        # Operation success component
        operation_success = 1.0 if result.get("success", False) else 0.0
        success_reward = operation_success * self.reward_weights["operation_success"]
        
        # Composite reward
        total_reward = qa_reward + integrity_reward + efficiency_reward + success_reward
        
        # Apply operation-specific bonuses/penalties
        if operation == GraphOperation.NOOP and len(result.get("extracted_facts", [])) > 0:
            total_reward -= 0.1  # Penalty for not acting when facts are available
        elif operation != GraphOperation.NOOP and len(result.get("extracted_facts", [])) == 0:
            total_reward -= 0.1  # Penalty for acting when no facts available
        
        return np.clip(total_reward, -1.0, 1.0)
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        
        if MEMORY_R1_AVAILABLE:
            status = self.memory_system.get_system_status()
            
            # Calculate efficiency based on operations per extraction
            total_ops = status["system_stats"]["total_operations"]
            total_extractions = status["system_stats"]["total_extractions"]
            
            if total_extractions > 0:
                ops_per_extraction = total_ops / total_extractions
                # Efficiency decreases as operations per extraction increases
                efficiency = max(0.0, 1.0 - (ops_per_extraction - 1.0) / 5.0)
            else:
                efficiency = 0.5
            
            return efficiency
        else:
            return 0.5  # Fallback
    
    def _calculate_graph_integrity(self) -> float:
        """Calculate graph integrity score"""
        
        if MEMORY_R1_AVAILABLE:
            validation = self.memory_system.validate_graph_consistency()
            
            if validation["overall_status"] == "valid":
                return 1.0
            elif validation["overall_status"] == "warning":
                return 0.7
            else:
                return 0.3
        else:
            return 0.5  # Fallback
    
    def get_trace_buffer(self) -> List[Dict[str, Any]]:
        """Get complete trace buffer for analysis"""
        return self.trace_buffer.copy()
    
    def export_episode_traces(self, output_path: str = None) -> str:
        """Export episode traces for external analysis"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"graph_env_traces_{timestamp}.json"
        
        export_data = {
            "environment_config": self.config,
            "total_episodes": self.current_episode,
            "trace_buffer": self.trace_buffer,
            "current_episode_traces": self.episode_traces,
            "dataset_size": len(self.dataset),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📄 Episode traces exported to {output_path}")
        return output_path
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """Get comprehensive environment statistics"""
        
        stats = {
            "environment_info": {
                "current_episode": self.current_episode,
                "current_turn": self.current_turn,
                "max_turns": self.max_turns,
                "dataset_size": len(self.dataset)
            },
            "trace_statistics": {
                "total_episodes": len(self.trace_buffer),
                "current_episode_length": len(self.episode_traces),
                "avg_episode_length": np.mean([ep["episode_length"] for ep in self.trace_buffer]) if self.trace_buffer else 0,
                "total_traces": sum(ep["episode_length"] for ep in self.trace_buffer) + len(self.episode_traces)
            },
            "reward_statistics": {
                "total_reward": sum(ep["episode_reward"] for ep in self.trace_buffer),
                "avg_episode_reward": np.mean([ep["episode_reward"] for ep in self.trace_buffer]) if self.trace_buffer else 0,
                "current_episode_reward": sum(trace.get("reward", 0) for trace in self.episode_traces)
            }
        }
        
        # Add Memory-R1 stats if available
        if MEMORY_R1_AVAILABLE:
            memory_stats = self.memory_system.get_system_status()
            stats["memory_system_stats"] = memory_stats
        
        return stats

# PPO/GRPO Integration Wrapper
class PPOGraphTrainer:
    """PPO trainer wrapper for graph memory environment"""
    
    def __init__(self, env: GraphMemoryEnv, config: Dict[str, Any] = None):
        self.env = env
        self.config = config or {}
        
        # PPO hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.entropy_coeff = config.get("entropy_coeff", 0.01)
        self.value_coeff = config.get("value_coeff", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        
        # Training state
        self.policy_network = self._initialize_policy_network()
        self.value_network = self._initialize_value_network()
        self.training_history = []
        
        print("🎯 PPO Graph Trainer initialized")
    
    def _initialize_policy_network(self) -> Dict[str, Any]:
        """Initialize policy network (simplified representation)"""
        return {
            "weights": np.random.normal(0, 0.1, (self.env.observation_dim, self.env.action_space_size)),
            "bias": np.zeros(self.env.action_space_size)
        }
    
    def _initialize_value_network(self) -> Dict[str, Any]:
        """Initialize value network (simplified representation)"""
        return {
            "weights": np.random.normal(0, 0.1, (self.env.observation_dim, 1)),
            "bias": np.zeros(1)
        }
    
    def train_episode(self) -> Dict[str, Any]:
        """Train one episode using PPO"""
        
        # Collect episode data
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        state = self.env.reset()
        episode_reward = 0.0
        
        for step in range(self.env.max_turns):
            # Get action from policy
            action, log_prob = self._select_action(state)
            value = self._estimate_value(state)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards)
        advantages = self._calculate_advantages(returns, values)
        
        # PPO update
        training_metrics = self._ppo_update(states, actions, returns, advantages, log_probs)
        
        # Record training episode
        episode_record = {
            "episode": len(self.training_history),
            "episode_reward": episode_reward,
            "episode_length": len(states),
            "training_metrics": training_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_history.append(episode_record)
        
        return episode_record
    
    def _select_action(self, state: EnvironmentState) -> Tuple[int, float]:
        """Select action using policy network"""
        
        # Simple policy network forward pass
        logits = np.dot(state.graph_features, self.policy_network["weights"]) + self.policy_network["bias"]
        probs = self._softmax(logits)
        
        # Sample action
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        
        return action, log_prob
    
    def _estimate_value(self, state: EnvironmentState) -> float:
        """Estimate state value using value network"""
        
        value = np.dot(state.graph_features, self.value_network["weights"]) + self.value_network["bias"]
        return float(value[0])
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def _calculate_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """Calculate discounted returns"""
        returns = []
        running_return = 0.0
        
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
        
        return returns
    
    def _calculate_advantages(self, returns: List[float], values: List[float]) -> List[float]:
        """Calculate advantages"""
        advantages = [ret - val for ret, val in zip(returns, values)]
        
        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages
    
    def _ppo_update(self, states: List[EnvironmentState], actions: List[int], 
                   returns: List[float], advantages: List[float], 
                   old_log_probs: List[float]) -> Dict[str, float]:
        """Perform PPO update (simplified implementation)"""
        
        # Calculate new log probabilities
        new_log_probs = []
        for state, action in zip(states, actions):
            _, log_prob = self._select_action(state)
            new_log_probs.append(log_prob)
        
        # Calculate probability ratios
        ratios = [np.exp(new_lp - old_lp) for new_lp, old_lp in zip(new_log_probs, old_log_probs)]
        
        # PPO clipped objective
        policy_losses = []
        for ratio, advantage in zip(ratios, advantages):
            clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            loss1 = ratio * advantage
            loss2 = clipped_ratio * advantage
            policy_loss = -min(loss1, loss2)
            policy_losses.append(policy_loss)
        
        # Simple gradient update (in practice, use proper optimizer)
        avg_policy_loss = np.mean(policy_losses)
        
        # Update policy weights (simplified)
        gradient_scale = self.learning_rate * avg_policy_loss
        self.policy_network["weights"] -= gradient_scale * 0.01 * np.random.normal(0, 0.1, self.policy_network["weights"].shape)
        
        return {
            "policy_loss": avg_policy_loss,
            "avg_ratio": np.mean(ratios),
            "avg_advantage": np.mean(advantages),
            "entropy": -np.mean(new_log_probs)  # Simplified entropy
        }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status"""
        
        return {
            "episodes_trained": len(self.training_history),
            "total_reward": sum(ep["episode_reward"] for ep in self.training_history),
            "avg_episode_reward": np.mean([ep["episode_reward"] for ep in self.training_history]) if self.training_history else 0,
            "recent_performance": {
                "last_5_avg_reward": np.mean([ep["episode_reward"] for ep in self.training_history[-5:]]) if len(self.training_history) >= 5 else 0,
                "reward_trend": "improving" if len(self.training_history) > 1 and self.training_history[-1]["episode_reward"] > self.training_history[0]["episode_reward"] else "stable"
            },
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "clip_epsilon": self.clip_epsilon,
                "entropy_coeff": self.entropy_coeff
            }
        }

# Utility functions
def create_graph_env(config: Dict[str, Any] = None) -> GraphMemoryEnv:
    """Create and initialize graph memory environment"""
    return GraphMemoryEnv(config)

def create_ppo_trainer(env: GraphMemoryEnv, config: Dict[str, Any] = None) -> PPOGraphTrainer:
    """Create PPO trainer for graph environment"""
    return PPOGraphTrainer(env, config)

def run_training_demo(num_episodes: int = 5) -> Dict[str, Any]:
    """Run a training demonstration"""
    
    print(f"🎮 Graph Environment Training Demo")
    print(f"Training {num_episodes} episodes...")
    
    # Create environment and trainer
    env = create_graph_env()
    trainer = create_ppo_trainer(env)
    
    # Training loop
    training_results = []
    
    for episode in range(num_episodes):
        episode_result = trainer.train_episode()
        training_results.append(episode_result)
        
        print(f"Episode {episode + 1}: Reward = {episode_result['episode_reward']:.3f}")
    
    # Get final status
    final_status = trainer.get_training_status()
    env_stats = env.get_environment_stats()
    
    print(f"\n✅ Training completed!")
    print(f"   Total episodes: {final_status['episodes_trained']}")
    print(f"   Average reward: {final_status['avg_episode_reward']:.3f}")
    print(f"   Trend: {final_status['recent_performance']['reward_trend']}")
    
    return {
        "training_results": training_results,
        "final_status": final_status,
        "environment_stats": env_stats
    }

if __name__ == "__main__":
    # Run demo
    demo_results = run_training_demo()
