#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-R1 Modular Extension System
Advanced memory management with semantic graph reasoning, provenance tracking,
trace buffer replay, and CI-evaluable hooks for the AI Research Agent.
"""

import json
import uuid
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphOperation(Enum):
    """Graph operation types for memory management"""
    ADD_NODE = "add_node"
    MERGE_EDGE = "merge_edge"
    DELETE_SUBGRAPH = "delete_subgraph"
    NOOP = "noop"

class MemoryEntryType(Enum):
    """Types of memory entries"""
    FACT = "fact"
    EVENT = "event"
    ENTITY = "entity"
    RELATION = "relation"
    CONCEPT = "concept"

@dataclass
class GraphTriple:
    """Semantic graph triple (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_turn: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ProvenanceMetadata:
    """Provenance tracking metadata for memory entries"""
    content: str
    source_turn: int
    update_chain: List[str]
    confidence_score: float
    created_at: datetime
    last_updated: datetime
    transformation_history: List[Dict[str, Any]]
    trustworthiness: float = 0.8

@dataclass
class MemoryEntry:
    """Enhanced memory entry with provenance and graph structure"""
    entry_id: str
    content: str
    entry_type: MemoryEntryType
    graph_fragment: Optional[Dict[str, Any]]
    provenance: ProvenanceMetadata
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class TraceRecord:
    """Record for trace buffer replay"""
    turn_id: int
    input_text: str
    extracted_facts: List[str]
    memory_returned: List[str]
    output_text: str
    reward: Optional[float]
    graph_operations: List[Dict[str, Any]]
    timestamp: datetime
    session_id: str

class GraphBuilder:
    """Converts extracted facts into semantic graphs using dependency parsing + entity linking"""
    
    def __init__(self):
        self.entity_cache = {}
        self.relation_patterns = {
            "is_a": ["is", "are", "was", "were"],
            "has": ["has", "have", "contains", "includes"],
            "located_in": ["in", "at", "located", "situated"],
            "causes": ["causes", "leads to", "results in"],
            "part_of": ["part of", "component of", "belongs to"]
        }
        
        logger.info("🔧 GraphBuilder initialized")
    
    def extract_triples(self, text: str, source_turn: int = 0) -> List[GraphTriple]:
        """Extract semantic triples from text"""
        triples = []
        
        # Simple pattern-based extraction (in production, use spaCy/NLTK)
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Extract entities and relations using simple patterns
            extracted_triples = self._pattern_based_extraction(sentence, source_turn)
            triples.extend(extracted_triples)
        
        logger.debug(f"Extracted {len(triples)} triples from text")
        return triples
    
    def _pattern_based_extraction(self, sentence: str, source_turn: int) -> List[GraphTriple]:
        """Pattern-based triple extraction"""
        triples = []
        sentence_lower = sentence.lower()
        
        # Simple subject-predicate-object patterns
        for relation, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if pattern in sentence_lower:
                    parts = sentence_lower.split(pattern, 1)
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        obj = parts[1].strip()
                        
                        if subject and obj:
                            # Clean up entities
                            subject = self._clean_entity(subject)
                            obj = self._clean_entity(obj)
                            
                            triple = GraphTriple(
                                subject=subject,
                                predicate=relation,
                                object=obj,
                                confidence=0.7,
                                source_turn=source_turn
                            )
                            triples.append(triple)
        
        return triples
    
    def _clean_entity(self, entity: str) -> str:
        """Clean and normalize entity names"""
        # Remove articles and common words
        stop_words = {"the", "a", "an", "this", "that", "these", "those"}
        words = entity.split()
        cleaned_words = [w for w in words if w.lower() not in stop_words]
        return " ".join(cleaned_words).strip()
    
    def build_graph_fragment(self, triples: List[GraphTriple]) -> Dict[str, Any]:
        """Build graph fragment from triples"""
        nodes = {}
        edges = []
        
        for triple in triples:
            # Add subject node
            if triple.subject not in nodes:
                nodes[triple.subject] = {
                    "id": triple.subject,
                    "type": "entity",
                    "confidence": triple.confidence,
                    "source_turn": triple.source_turn
                }
            
            # Add object node
            if triple.object not in nodes:
                nodes[triple.object] = {
                    "id": triple.object,
                    "type": "entity",
                    "confidence": triple.confidence,
                    "source_turn": triple.source_turn
                }
            
            # Add edge
            edge = {
                "source": triple.subject,
                "target": triple.object,
                "relation": triple.predicate,
                "confidence": triple.confidence,
                "source_turn": triple.source_turn,
                "timestamp": triple.timestamp.isoformat()
            }
            edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "created_at": datetime.now().isoformat(),
            "triple_count": len(triples)
        }

class GraphMemoryBank:
    """Stores evolving graph state with efficient operations"""
    
    def __init__(self, storage_path: str = "memory_r1_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # NetworkX graph for structure
        self.graph = nx.MultiDiGraph()
        
        # Memory storage
        self.memory_entries: Dict[str, MemoryEntry] = {}
        self.graph_fragments: Dict[str, Dict[str, Any]] = {}
        
        # Indexing for fast retrieval
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.relation_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.operation_count = 0
        self.last_updated = datetime.now()
        
        logger.info("🏦 GraphMemoryBank initialized")
    
    def add_memory_entry(self, entry: MemoryEntry) -> str:
        """Add memory entry with graph fragment"""
        self.memory_entries[entry.entry_id] = entry
        
        if entry.graph_fragment:
            self.graph_fragments[entry.entry_id] = entry.graph_fragment
            self._integrate_graph_fragment(entry.entry_id, entry.graph_fragment)
        
        # Update indices
        self._update_indices(entry)
        
        self.operation_count += 1
        self.last_updated = datetime.now()
        
        logger.debug(f"Added memory entry: {entry.entry_id}")
        return entry.entry_id
    
    def _integrate_graph_fragment(self, entry_id: str, fragment: Dict[str, Any]):
        """Integrate graph fragment into main graph"""
        nodes = fragment.get("nodes", {})
        edges = fragment.get("edges", [])
        
        # Add nodes
        for node_id, node_data in nodes.items():
            if self.graph.has_node(node_id):
                # Merge node data
                existing_data = self.graph.nodes[node_id]
                existing_data.update(node_data)
            else:
                self.graph.add_node(node_id, **node_data)
            
            # Update entity index
            self.entity_index[node_id].add(entry_id)
        
        # Add edges
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            relation = edge["relation"]
            
            self.graph.add_edge(
                source, target,
                relation=relation,
                confidence=edge["confidence"],
                source_turn=edge["source_turn"],
                entry_id=entry_id
            )
            
            # Update relation index
            self.relation_index[relation].add(entry_id)
    
    def _update_indices(self, entry: MemoryEntry):
        """Update search indices"""
        # Temporal index
        date_key = entry.provenance.created_at.strftime("%Y-%m-%d")
        self.temporal_index[date_key].append(entry.entry_id)
        
        # Tag index (if needed)
        for tag in entry.tags:
            self.entity_index[tag].add(entry.entry_id)
    
    def query_by_entity(self, entity: str, max_results: int = 10) -> List[MemoryEntry]:
        """Query memory entries by entity"""
        entry_ids = self.entity_index.get(entity, set())
        
        results = []
        for entry_id in list(entry_ids)[:max_results]:
            if entry_id in self.memory_entries:
                results.append(self.memory_entries[entry_id])
        
        return results
    
    def query_by_relation(self, relation: str, max_results: int = 10) -> List[MemoryEntry]:
        """Query memory entries by relation"""
        entry_ids = self.relation_index.get(relation, set())
        
        results = []
        for entry_id in list(entry_ids)[:max_results]:
            if entry_id in self.memory_entries:
                results.append(self.memory_entries[entry_id])
        
        return results
    
    def get_connected_entities(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get entities connected to given entity"""
        if not self.graph.has_node(entity):
            return {"entities": [], "relations": []}
        
        connected = {"entities": [], "relations": []}
        
        # BFS to find connected entities
        visited = set()
        queue = deque([(entity, 0)])
        
        while queue:
            current_entity, depth = queue.popleft()
            
            if depth > max_depth or current_entity in visited:
                continue
            
            visited.add(current_entity)
            
            # Get neighbors
            for neighbor in self.graph.neighbors(current_entity):
                if neighbor not in visited:
                    connected["entities"].append(neighbor)
                    
                    # Get edge data
                    edge_data = self.graph.get_edge_data(current_entity, neighbor)
                    if edge_data:
                        for edge_key, edge_attrs in edge_data.items():
                            connected["relations"].append({
                                "source": current_entity,
                                "target": neighbor,
                                "relation": edge_attrs.get("relation", "unknown"),
                                "confidence": edge_attrs.get("confidence", 0.5)
                            })
                    
                    if depth + 1 <= max_depth:
                        queue.append((neighbor, depth + 1))
        
        return connected
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "memory_entries": len(self.memory_entries),
            "graph_fragments": len(self.graph_fragments),
            "operation_count": self.operation_count,
            "last_updated": self.last_updated.isoformat(),
            "entity_types": len(self.entity_index),
            "relation_types": len(self.relation_index)
        }

class GraphRLPolicy:
    """Trained via PPO/GRPO to manage graph operations"""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.policy_weights = {}
        self.operation_history = []
        self.reward_history = []
        
        # Simple heuristic weights (in production, use neural network)
        self.operation_weights = {
            GraphOperation.ADD_NODE: 0.8,
            GraphOperation.MERGE_EDGE: 0.7,
            GraphOperation.DELETE_SUBGRAPH: 0.3,
            GraphOperation.NOOP: 0.1
        }
        
        logger.info("🤖 GraphRLPolicy initialized")
    
    def select_operation(self, context: Dict[str, Any]) -> Tuple[GraphOperation, Dict[str, Any]]:
        """Select graph operation based on context"""
        # Extract context features
        new_triples = context.get("new_triples", [])
        existing_entities = context.get("existing_entities", set())
        confidence_threshold = context.get("confidence_threshold", 0.5)
        
        # Decision logic (simplified)
        if not new_triples:
            return GraphOperation.NOOP, {}
        
        # Check if we should add new nodes
        new_entities = set()
        for triple in new_triples:
            if hasattr(triple, 'subject') and triple.subject not in existing_entities:
                new_entities.add(triple.subject)
            if hasattr(triple, 'object') and triple.object not in existing_entities:
                new_entities.add(triple.object)
        
        if new_entities:
            operation = GraphOperation.ADD_NODE
            params = {"entities": list(new_entities)}
        else:
            # Merge edges for existing entities
            operation = GraphOperation.MERGE_EDGE
            params = {"triples": [asdict(t) for t in new_triples]}
        
        # Record decision
        decision = {
            "operation": operation,
            "params": params,
            "context": context,
            "timestamp": datetime.now()
        }
        self.operation_history.append(decision)
        
        return operation, params
    
    def update_policy(self, reward: float):
        """Update policy based on reward"""
        if self.operation_history:
            last_operation = self.operation_history[-1]["operation"]
            
            # Simple policy update (in production, use PPO/GRPO)
            current_weight = self.operation_weights.get(last_operation, 0.5)
            
            # Update weight based on reward
            if reward > 0:
                new_weight = min(1.0, current_weight + self.learning_rate * reward)
            else:
                new_weight = max(0.0, current_weight + self.learning_rate * reward)
            
            self.operation_weights[last_operation] = new_weight
            self.reward_history.append(reward)
        
        logger.debug(f"Policy updated with reward: {reward}")
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get policy statistics"""
        return {
            "operation_weights": dict(self.operation_weights),
            "total_decisions": len(self.operation_history),
            "average_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "recent_rewards": self.reward_history[-10:] if self.reward_history else []
        }

class ProvenanceTracker:
    """Logs source and transformation history"""
    
    def __init__(self):
        self.provenance_log = []
        self.source_mapping = {}
        
        logger.info("📋 ProvenanceTracker initialized")
    
    def create_provenance(self, content: str, source_turn: int, 
                         confidence_score: float = 0.8) -> ProvenanceMetadata:
        """Create initial provenance metadata"""
        provenance = ProvenanceMetadata(
            content=content,
            source_turn=source_turn,
            update_chain=[f"initial_creation_{source_turn}"],
            confidence_score=confidence_score,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            transformation_history=[{
                "operation": "create",
                "timestamp": datetime.now().isoformat(),
                "source_turn": source_turn
            }]
        )
        
        return provenance
    
    def update_provenance(self, provenance: ProvenanceMetadata, 
                         operation: str, source_turn: int,
                         confidence_delta: float = 0.0) -> ProvenanceMetadata:
        """Update provenance chain"""
        # Update chain
        update_id = f"{operation}_{source_turn}_{datetime.now().timestamp()}"
        provenance.update_chain.append(update_id)
        
        # Update confidence
        provenance.confidence_score = max(0.0, min(1.0, 
            provenance.confidence_score + confidence_delta))
        
        # Add transformation record
        transformation = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "source_turn": source_turn,
            "confidence_delta": confidence_delta,
            "update_id": update_id
        }
        provenance.transformation_history.append(transformation)
        
        # Update timestamps
        provenance.last_updated = datetime.now()
        
        # Log the update
        self.provenance_log.append({
            "update_id": update_id,
            "operation": operation,
            "source_turn": source_turn,
            "timestamp": datetime.now().isoformat()
        })
        
        return provenance
    
    def validate_provenance_chain(self, provenance: ProvenanceMetadata) -> bool:
        """Validate provenance chain integrity"""
        if not provenance.update_chain:
            return False
        
        # Check chain consistency
        if len(provenance.update_chain) != len(provenance.transformation_history):
            return False
        
        # Check temporal ordering
        timestamps = [t["timestamp"] for t in provenance.transformation_history]
        sorted_timestamps = sorted(timestamps)
        
        return timestamps == sorted_timestamps
    
    def get_provenance_summary(self, provenance: ProvenanceMetadata) -> Dict[str, Any]:
        """Get provenance summary"""
        return {
            "source_turn": provenance.source_turn,
            "confidence_score": provenance.confidence_score,
            "trustworthiness": provenance.trustworthiness,
            "update_count": len(provenance.update_chain),
            "age_hours": (datetime.now() - provenance.created_at).total_seconds() / 3600,
            "last_updated": provenance.last_updated.isoformat(),
            "transformation_count": len(provenance.transformation_history)
        }

class ConfidenceScorer:
    """Uses heuristics or learned model to assign trust scores"""
    
    def __init__(self):
        self.scoring_weights = {
            "source_reliability": 0.3,
            "temporal_freshness": 0.2,
            "cross_validation": 0.2,
            "consistency": 0.15,
            "completeness": 0.15
        }
        
        logger.info("🎯 ConfidenceScorer initialized")
    
    def calculate_confidence(self, entry: MemoryEntry, 
                           context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for memory entry"""
        scores = {}
        
        # Source reliability
        source_turn = entry.provenance.source_turn
        scores["source_reliability"] = min(1.0, 0.5 + (source_turn / 100.0))
        
        # Temporal freshness
        age_hours = (datetime.now() - entry.provenance.created_at).total_seconds() / 3600
        scores["temporal_freshness"] = max(0.1, 1.0 - (age_hours / (24 * 7)))  # Week decay
        
        # Cross validation (based on update chain length)
        update_count = len(entry.provenance.update_chain)
        scores["cross_validation"] = min(1.0, 0.3 + (update_count * 0.1))
        
        # Consistency (based on confidence score stability)
        scores["consistency"] = entry.provenance.confidence_score
        
        # Completeness (based on content length and graph fragment)
        content_score = min(1.0, len(entry.content) / 200.0)
        graph_score = 1.0 if entry.graph_fragment else 0.5
        scores["completeness"] = (content_score + graph_score) / 2.0
        
        # Weighted combination
        confidence = sum(
            scores[factor] * weight 
            for factor, weight in self.scoring_weights.items()
        )
        
        return max(0.0, min(1.0, confidence))
    
    def update_trustworthiness(self, entry: MemoryEntry, 
                              validation_result: bool) -> float:
        """Update trustworthiness based on validation"""
        current_trust = entry.provenance.trustworthiness
        
        if validation_result:
            # Increase trust
            new_trust = min(1.0, current_trust + 0.1)
        else:
            # Decrease trust
            new_trust = max(0.0, current_trust - 0.2)
        
        entry.provenance.trustworthiness = new_trust
        return new_trust

class ValidatorHook:
    """Enforces update constraints (e.g., no overwrite without source match)"""
    
    def __init__(self):
        self.validation_rules = {
            "source_match_required": True,
            "confidence_threshold": 0.3,
            "max_update_frequency": 10,  # per hour
            "require_provenance": True
        }
        
        self.violation_log = []
        
        logger.info("🔒 ValidatorHook initialized")
    
    def validate_update(self, existing_entry: Optional[MemoryEntry], 
                       new_content: str, source_turn: int) -> Tuple[bool, str]:
        """Validate memory update operation"""
        
        if not existing_entry:
            # New entry - always allowed
            return True, "New entry creation allowed"
        
        # Check source match requirement
        if self.validation_rules["source_match_required"]:
            if existing_entry.provenance.source_turn != source_turn:
                # Allow if confidence is low
                if existing_entry.provenance.confidence_score < self.validation_rules["confidence_threshold"]:
                    return True, "Low confidence entry can be updated"
                else:
                    violation = {
                        "type": "source_mismatch",
                        "existing_source": existing_entry.provenance.source_turn,
                        "new_source": source_turn,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.violation_log.append(violation)
                    return False, "Source mismatch - update rejected"
        
        # Check update frequency
        last_update = existing_entry.provenance.last_updated
        time_since_update = (datetime.now() - last_update).total_seconds() / 3600
        
        if time_since_update < (1.0 / self.validation_rules["max_update_frequency"]):
            violation = {
                "type": "update_frequency",
                "time_since_update": time_since_update,
                "timestamp": datetime.now().isoformat()
            }
            self.violation_log.append(violation)
            return False, "Update frequency limit exceeded"
        
        return True, "Update validation passed"
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        violation_types = defaultdict(int)
        for violation in self.violation_log:
            violation_types[violation["type"]] += 1
        
        return {
            "total_violations": len(self.violation_log),
            "violation_types": dict(violation_types),
            "validation_rules": self.validation_rules,
            "recent_violations": self.violation_log[-10:]
        }

class TraceBuffer:
    """Circular buffer storing recent agent interactions"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.trace_index = {}
        
        logger.info(f"📊 TraceBuffer initialized with max_size={max_size}")
    
    def add_trace(self, trace: TraceRecord):
        """Add trace record to buffer"""
        self.buffer.append(trace)
        self.trace_index[trace.turn_id] = trace
        
        # Clean up old indices if buffer is full
        if len(self.buffer) == self.max_size:
            # Remove oldest trace from index
            oldest_trace = self.buffer[0]
            if oldest_trace.turn_id in self.trace_index:
                del self.trace_index[oldest_trace.turn_id]
    
    def get_trace_range(self, start_turn: int, end_turn: int) -> List[TraceRecord]:
        """Get traces in turn range"""
        traces = []
        for trace in self.buffer:
            if start_turn <= trace.turn_id <= end_turn:
                traces.append(trace)
        
        return sorted(traces, key=lambda t: t.turn_id)
    
    def get_recent_traces(self, count: int = 10) -> List[TraceRecord]:
        """Get most recent traces"""
        return list(self.buffer)[-count:]
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.buffer:
            return {"empty": True}
        
        rewards = [t.reward for t in self.buffer if t.reward is not None]
        
        return {
            "total_traces": len(self.buffer),
            "buffer_utilization": len(self.buffer) / self.max_size,
            "turn_range": {
                "min": min(t.turn_id for t in self.buffer),
                "max": max(t.turn_id for t in self.buffer)
            },
            "average_reward": np.mean(rewards) if rewards else None,
            "session_count": len(set(t.session_id for t in self.buffer))
        }

class ReplayEngine:
    """Reconstructs memory state and agent decisions over time"""
    
    def __init__(self, memory_bank: GraphMemoryBank, trace_buffer: TraceBuffer):
        self.memory_bank = memory_bank
        self.trace_buffer = trace_buffer
        
        logger.info("🔄 ReplayEngine initialized")
    
    def replay_memory_evolution(self, start_turn: int, end_turn: int) -> Dict[str, Any]:
        """Replay memory evolution over turn range"""
        traces = self.trace_buffer.get_trace_range(start_turn, end_turn)
        
        evolution = {
            "start_turn": start_turn,
            "end_turn": end_turn,
            "trace_count": len(traces),
            "memory_operations": [],
            "graph_changes": [],
            "decision_points": []
        }
        
        for trace in traces:
            # Analyze memory operations
            for op in trace.graph_operations:
                evolution["memory_operations"].append({
                    "turn_id": trace.turn_id,
                    "operation": op,
                    "timestamp": trace.timestamp.isoformat()
                })
            
            # Track decision points
            if trace.reward is not None:
                evolution["decision_points"].append({
                    "turn_id": trace.turn_id,
                    "input": trace.input_text[:100] + "..." if len(trace.input_text) > 100 else trace.input_text,
                    "output": trace.output_text[:100] + "..." if len(trace.output_text) > 100 else trace.output_text,
                    "reward": trace.reward,
                    "memory_used": len(trace.memory_returned)
                })
        
        return evolution
    
    def reconstruct_memory_state(self, target_turn: int) -> Dict[str, Any]:
        """Reconstruct memory state at specific turn"""
        traces = self.trace_buffer.get_trace_range(0, target_turn)
        
        # Simulate memory state reconstruction
        simulated_entries = {}
        operation_count = 0
        
        for trace in traces:
            for op in trace.graph_operations:
                operation_count += 1
                
                if op.get("type") == "add_memory":
                    entry_id = op.get("entry_id", f"sim_{operation_count}")
                    simulated_entries[entry_id] = {
                        "content": op.get("content", ""),
                        "turn_created": trace.turn_id,
                        "timestamp": trace.timestamp.isoformat()
                    }
        
        return {
            "target_turn": target_turn,
            "memory_entries": len(simulated_entries),
            "operation_count": operation_count,
            "entries": simulated_entries
        }

class RewardAttributor:
    """Assigns delayed rewards to memory ops based on downstream QA success"""
    
    def __init__(self):
        self.attribution_history = []
        self.reward_mapping = {}
        
        logger.info("🎁 RewardAttributor initialized")
    
    def attribute_reward(self, success_turn: int, reward: float, 
                        lookback_window: int = 5) -> Dict[str, Any]:
        """Attribute reward to recent memory operations"""
        
        # Find relevant traces in lookback window
        start_turn = max(0, success_turn - lookback_window)
        
        attribution = {
            "success_turn": success_turn,
            "reward": reward,
            "lookback_window": lookback_window,
            "attributed_operations": []
        }
        
        # Simple attribution: distribute reward based on recency
        for turn_offset in range(lookback_window):
            turn_id = success_turn - turn_offset - 1
            if turn_id >= 0:
                # Calculate attribution weight (more recent = higher weight)
                weight = (lookback_window - turn_offset) / lookback_window
                attributed_reward = reward * weight * 0.1  # Scale down
                
                attribution["attributed_operations"].append({
                    "turn_id": turn_id,
                    "attributed_reward": attributed_reward,
                    "weight": weight
                })
                
                # Store in mapping
                if turn_id not in self.reward_mapping:
                    self.reward_mapping[turn_id] = []
                self.reward_mapping[turn_id].append(attributed_reward)
        
        self.attribution_history.append(attribution)
        return attribution
    
    def get_turn_reward_summary(self, turn_id: int) -> Dict[str, Any]:
        """Get reward summary for specific turn"""
        rewards = self.reward_mapping.get(turn_id, [])
        
        return {
            "turn_id": turn_id,
            "total_attributed_reward": sum(rewards),
            "attribution_count": len(rewards),
            "average_reward": np.mean(rewards) if rewards else 0.0,
            "individual_rewards": rewards
        }

class MemoryR1System:
    """Main Memory-R1 system integrating all modules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize modules
        self.graph_builder = GraphBuilder()
        self.memory_bank = GraphMemoryBank()
        self.rl_policy = GraphRLPolicy()
        self.provenance_tracker = ProvenanceTracker()
        self.confidence_scorer = ConfidenceScorer()
        self.validator_hook = ValidatorHook()
        self.trace_buffer = TraceBuffer(max_size=self.config.get("trace_buffer_size", 1000))
        self.replay_engine = ReplayEngine(self.memory_bank, self.trace_buffer)
        self.reward_attributor = RewardAttributor()
        
        # System state
        self.current_turn = 0
        self.session_id = str(uuid.uuid4())
        
        logger.info("🚀 Memory-R1 System initialized")
    
    def process_input(self, input_text: str, turn_id: int = None) -> Dict[str, Any]:
        """Process input through Memory-R1 pipeline"""
        
        if turn_id is None:
            turn_id = self.current_turn
            self.current_turn += 1
        
        # Step 1: Extract facts and build graph triples
        triples = self.graph_builder.extract_triples(input_text, turn_id)
        
        # Step 2: Determine graph operations using RL policy
        existing_entities = set(self.memory_bank.graph.nodes())
        context = {
            "new_triples": triples,
            "existing_entities": existing_entities,
            "turn_id": turn_id
        }
        
        operation, params = self.rl_policy.select_operation(context)
        
        # Step 3: Create memory entries with provenance
        memory_entries = []
        graph_operations = []
        
        if operation != GraphOperation.NOOP and triples:
            for i, triple in enumerate(triples):
                # Create provenance
                provenance = self.provenance_tracker.create_provenance(
                    content=f"{triple.subject} {triple.predicate} {triple.object}",
                    source_turn=turn_id,
                    confidence_score=triple.confidence
                )
                
                # Build graph fragment
                graph_fragment = self.graph_builder.build_graph_fragment([triple])
                
                # Create memory entry
                entry = MemoryEntry(
                    entry_id=str(uuid.uuid4()),
                    content=f"{triple.subject} {triple.predicate} {triple.object}",
                    entry_type=MemoryEntryType.FACT,
                    graph_fragment=graph_fragment,
                    provenance=provenance,
                    tags=[triple.subject, triple.object]
                )
                
                # Validate and add to memory bank
                validation_result, validation_msg = self.validator_hook.validate_update(
                    existing_entry=None,  # New entry
                    new_content=entry.content,
                    source_turn=turn_id
                )
                
                if validation_result:
                    entry_id = self.memory_bank.add_memory_entry(entry)
                    memory_entries.append(entry)
                    
                    graph_operations.append({
                        "type": "add_memory",
                        "entry_id": entry_id,
                        "content": entry.content,
                        "operation": operation.value
                    })
        
        # Step 4: Query relevant memories for response
        relevant_memories = self._query_relevant_memories(input_text)
        
        # Step 5: Create trace record
        trace = TraceRecord(
            turn_id=turn_id,
            input_text=input_text,
            extracted_facts=[f"{t.subject} {t.predicate} {t.object}" for t in triples],
            memory_returned=[m.content for m in relevant_memories],
            output_text="",  # Will be filled by downstream components
            reward=None,  # Will be filled later
            graph_operations=graph_operations,
            timestamp=datetime.now(),
            session_id=self.session_id
        )
        
        self.trace_buffer.add_trace(trace)
        
        return {
            "turn_id": turn_id,
            "extracted_triples": len(triples),
            "graph_operation": operation.value,
            "memory_entries_created": len(memory_entries),
            "relevant_memories": len(relevant_memories),
            "memory_content": [m.content for m in relevant_memories],
            "trace_id": trace.turn_id
        }
    
    def _query_relevant_memories(self, query: str, max_results: int = 5) -> List[MemoryEntry]:
        """Query relevant memories for input"""
        # Simple keyword-based retrieval (in production, use embeddings)
        query_words = set(query.lower().split())
        
        scored_entries = []
        
        for entry in self.memory_bank.memory_entries.values():
            # Calculate relevance score
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words.intersection(entry_words))
            
            if overlap > 0:
                # Weight by confidence and recency
                confidence = self.confidence_scorer.calculate_confidence(entry)
                age_hours = (datetime.now() - entry.provenance.created_at).total_seconds() / 3600
                recency_weight = max(0.1, 1.0 - (age_hours / (24 * 7)))  # Week decay
                
                score = overlap * confidence * recency_weight
                scored_entries.append((score, entry))
        
        # Sort by score and return top results
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for score, entry in scored_entries[:max_results]]
    
    def update_reward(self, turn_id: int, reward: float):
        """Update reward for specific turn"""
        # Update RL policy
        self.rl_policy.update_policy(reward)
        
        # Update trace record
        if turn_id in self.trace_buffer.trace_index:
            self.trace_buffer.trace_index[turn_id].reward = reward
        
        # Attribute reward to recent operations
        attribution = self.reward_attributor.attribute_reward(turn_id, reward)
        
        logger.info(f"Updated reward for turn {turn_id}: {reward}")
        return attribution
    
    # CI-Evaluable Hooks
    def validate_graph_state(self) -> Dict[str, Any]:
        """Validate current graph state"""
        stats = self.memory_bank.get_graph_statistics()
        
        validation = {
            "valid": True,
            "issues": [],
            "statistics": stats
        }
        
        # Check for basic consistency
        if stats["total_nodes"] == 0 and stats["memory_entries"] > 0:
            validation["valid"] = False
            validation["issues"].append("Memory entries exist but no graph nodes")
        
        if stats["total_edges"] > stats["total_nodes"] * (stats["total_nodes"] - 1):
            validation["valid"] = False
            validation["issues"].append("Too many edges for number of nodes")
        
        return validation
    
    def check_provenance_integrity(self) -> Dict[str, Any]:
        """Check provenance integrity across all entries"""
        integrity = {
            "valid": True,
            "total_entries": len(self.memory_bank.memory_entries),
            "valid_entries": 0,
            "invalid_entries": 0,
            "issues": []
        }
        
        for entry_id, entry in self.memory_bank.memory_entries.items():
            is_valid = self.provenance_tracker.validate_provenance_chain(entry.provenance)
            
            if is_valid:
                integrity["valid_entries"] += 1
            else:
                integrity["invalid_entries"] += 1
                integrity["issues"].append(f"Invalid provenance for entry {entry_id}")
        
        if integrity["invalid_entries"] > 0:
            integrity["valid"] = False
        
        return integrity
    
    def replay_trace(self, start_turn: int, end_turn: int) -> Dict[str, Any]:
        """Replay trace for debugging and analysis"""
        return self.replay_engine.replay_memory_evolution(start_turn, end_turn)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "memory_bank": self.memory_bank.get_graph_statistics(),
            "trace_buffer": self.trace_buffer.get_buffer_statistics(),
            "rl_policy": self.rl_policy.get_policy_statistics(),
            "validator": self.validator_hook.get_validation_statistics(),
            "graph_validation": self.validate_graph_state(),
            "provenance_integrity": self.check_provenance_integrity()
        }

# Convenience functions for integration
def create_memory_r1_system(config: Dict[str, Any] = None) -> MemoryR1System:
    """Create and initialize Memory-R1 system"""
    return MemoryR1System(config)

def integrate_with_research_agent(memory_system: MemoryR1System, 
                                 research_agent) -> Dict[str, Any]:
    """Integrate Memory-R1 with AI Research Agent"""
    
    integration_stats = {
        "memory_system_id": memory_system.session_id,
        "integration_timestamp": datetime.now().isoformat(),
        "modules_integrated": [
            "semantic_graph_reasoning",
            "provenance_validation", 
            "trace_buffer_replay",
            "ci_evaluable_hooks"
        ]
    }
    
    # Hook into research agent's memory operations
    # (Implementation depends on research agent architecture)
    
    logger.info("🔗 Memory-R1 integrated with AI Research Agent")
    return integration_stats