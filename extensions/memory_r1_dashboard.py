#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-R1 Interactive Dashboard
Designed for contributors, researchers, and policy auditors to inspect memory evolution and agent behavior.
Implements the dashboard layout: Graph Memory View, Provenance Explorer, Trace Replay Panel, QA Outcome Viewer, Policy Metrics
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import io
import base64

class MemoryR1Dashboard:
    """Interactive dashboard for Memory-R1 system inspection and analysis"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.dashboard_state = {
            "current_view": "graph_memory",
            "selected_node": None,
            "selected_trace": None,
            "time_range": {"start": 1, "end": 10},
            "filters": {
                "confidence_threshold": 0.5,
                "show_orphans": True,
                "operation_types": ["all"]
            }
        }
        
        print("📊 Memory-R1 Dashboard initialized")
    
    def generate_graph_memory_view(self) -> Dict[str, Any]:
        """Generate Graph Memory View: Node-link diagram, entity types, edge predicates"""
        
        view_data = {
            "view_type": "graph_memory",
            "timestamp": datetime.now().isoformat(),
            "graph_structure": {
                "nodes": [],
                "edges": [],
                "clusters": [],
                "statistics": {}
            },
            "entity_types": {},
            "edge_predicates": {},
            "visualization_data": {}
        }
        
        try:
            graph = self.memory_system.graph_memory.graph
            
            # Extract nodes with metadata
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                
                node_info = {
                    "id": node_id,
                    "label": node_id,
                    "type": node_data.get("entity_type", "unknown"),
                    "confidence": node_data.get("confidence", 0.5),
                    "first_seen": node_data.get("first_seen", "unknown"),
                    "degree": graph.degree(node_id),
                    "position": self._calculate_node_position(node_id, graph)
                }
                
                view_data["graph_structure"]["nodes"].append(node_info)
                
                # Count entity types
                entity_type = node_info["type"]
                view_data["entity_types"][entity_type] = view_data["entity_types"].get(entity_type, 0) + 1
            
            # Extract edges with relationships
            for source, target, edge_data in graph.edges(data=True):
                edge_info = {
                    "source": source,
                    "target": target,
                    "relation": edge_data.get("relation", "unknown"),
                    "confidence": edge_data.get("confidence", 0.5),
                    "source_turn": edge_data.get("source_turn", 0),
                    "weight": edge_data.get("confidence", 0.5)
                }
                
                view_data["graph_structure"]["edges"].append(edge_info)
                
                # Count edge predicates
                predicate = edge_info["relation"]
                view_data["edge_predicates"][predicate] = view_data["edge_predicates"].get(predicate, 0) + 1
            
            # Calculate graph statistics
            view_data["graph_structure"]["statistics"] = {
                "total_nodes": len(view_data["graph_structure"]["nodes"]),
                "total_edges": len(view_data["graph_structure"]["edges"]),
                "connected_components": nx.number_weakly_connected_components(graph),
                "average_degree": np.mean([node["degree"] for node in view_data["graph_structure"]["nodes"]]) if view_data["graph_structure"]["nodes"] else 0,
                "density": nx.density(graph),
                "clustering_coefficient": nx.average_clustering(graph.to_undirected()) if len(graph.nodes) > 0 else 0
            }
            
            # Detect clusters/communities
            if len(graph.nodes) > 2:
                try:
                    undirected = graph.to_undirected()
                    communities = nx.community.greedy_modularity_communities(undirected)
                    
                    for i, community in enumerate(communities):
                        cluster_info = {
                            "id": f"cluster_{i}",
                            "nodes": list(community),
                            "size": len(community),
                            "density": self._calculate_cluster_density(community, graph)
                        }
                        view_data["graph_structure"]["clusters"].append(cluster_info)
                except:
                    pass  # Community detection failed
            
            # Generate visualization data
            view_data["visualization_data"] = self._generate_graph_visualization(
                view_data["graph_structure"]["nodes"],
                view_data["graph_structure"]["edges"]
            )
        
        except Exception as e:
            view_data["error"] = str(e)
        
        return view_data
    
    def generate_provenance_explorer(self, reference_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate Provenance Explorer: Source turn, update chain, confidence score, timestamps"""
        
        explorer_data = {
            "view_type": "provenance_explorer",
            "timestamp": datetime.now().isoformat(),
            "selected_reference": reference_id,
            "provenance_records": [],
            "update_chains": [],
            "confidence_timeline": [],
            "source_analysis": {}
        }
        
        try:
            provenance_tracker = self.memory_system.provenance_tracker
            
            if reference_id and reference_id in provenance_tracker.provenance_records:
                # Single reference analysis
                provenance = provenance_tracker.provenance_records[reference_id]
                
                explorer_data["provenance_records"] = [{
                    "entry_id": provenance.entry_id,
                    "content": provenance.content,
                    "source_turn": provenance.source_turn,
                    "confidence_score": provenance.confidence_score,
                    "trustworthiness": provenance.trustworthiness,
                    "created_at": provenance.created_at.isoformat(),
                    "last_updated": provenance.last_updated.isoformat(),
                    "validation_status": provenance.validation_status,
                    "update_chain_length": len(provenance.update_chain),
                    "transformation_count": len(provenance.transformation_history)
                }]
                
                # Update chain visualization
                explorer_data["update_chains"] = [{
                    "reference_id": reference_id,
                    "chain": provenance.update_chain,
                    "transformations": provenance.transformation_history
                }]
                
                # Confidence timeline
                timeline = []
                for i, transform in enumerate(provenance.transformation_history):
                    timeline.append({
                        "step": i,
                        "timestamp": transform["timestamp"],
                        "operation": transform["operation"],
                        "confidence": transform.get("confidence", provenance.confidence_score)
                    })
                
                explorer_data["confidence_timeline"] = timeline
            
            else:
                # All records overview
                for entry_id, provenance in provenance_tracker.provenance_records.items():
                    record_summary = {
                        "entry_id": entry_id,
                        "content_preview": provenance.content[:100] + "..." if len(provenance.content) > 100 else provenance.content,
                        "source_turn": provenance.source_turn,
                        "confidence_score": provenance.confidence_score,
                        "trustworthiness": provenance.trustworthiness,
                        "age_hours": (datetime.now() - provenance.created_at).total_seconds() / 3600,
                        "update_count": len(provenance.transformation_history),
                        "validation_status": provenance.validation_status
                    }
                    
                    explorer_data["provenance_records"].append(record_summary)
                
                # Source analysis
                source_turns = [p.source_turn for p in provenance_tracker.provenance_records.values()]
                confidence_scores = [p.confidence_score for p in provenance_tracker.provenance_records.values()]
                
                explorer_data["source_analysis"] = {
                    "total_records": len(provenance_tracker.provenance_records),
                    "source_turn_range": {
                        "min": min(source_turns) if source_turns else 0,
                        "max": max(source_turns) if source_turns else 0
                    },
                    "confidence_distribution": {
                        "mean": np.mean(confidence_scores) if confidence_scores else 0,
                        "std": np.std(confidence_scores) if confidence_scores else 0,
                        "min": min(confidence_scores) if confidence_scores else 0,
                        "max": max(confidence_scores) if confidence_scores else 0
                    },
                    "validation_status_counts": self._count_validation_statuses(provenance_tracker.provenance_records)
                }
        
        except Exception as e:
            explorer_data["error"] = str(e)
        
        return explorer_data
    
    def generate_trace_replay_panel(self, start_turn: int = None, end_turn: int = None) -> Dict[str, Any]:
        """Generate Trace Replay Panel: Timeline slider, action log, reward attribution"""
        
        if start_turn is None:
            start_turn = self.dashboard_state["time_range"]["start"]
        if end_turn is None:
            end_turn = self.dashboard_state["time_range"]["end"]
        
        panel_data = {
            "view_type": "trace_replay",
            "timestamp": datetime.now().isoformat(),
            "time_range": {"start": start_turn, "end": end_turn},
            "timeline_data": [],
            "action_log": [],
            "reward_attribution": {},
            "replay_controls": {
                "total_turns": len(self.memory_system.trace_buffer.traces),
                "available_range": {
                    "min": min([t.turn_id for t in self.memory_system.trace_buffer.traces]) if self.memory_system.trace_buffer.traces else 1,
                    "max": max([t.turn_id for t in self.memory_system.trace_buffer.traces]) if self.memory_system.trace_buffer.traces else 1
                },
                "playback_speed": 1.0,
                "current_position": start_turn
            }
        }
        
        try:
            # Get traces in range
            traces = self.memory_system.trace_buffer.get_traces_by_turn_range(start_turn, end_turn)
            
            # Generate timeline data
            for trace in traces:
                timeline_point = {
                    "turn_id": trace.turn_id,
                    "timestamp": trace.timestamp.isoformat(),
                    "input_preview": trace.input_text[:50] + "..." if len(trace.input_text) > 50 else trace.input_text,
                    "facts_extracted": len(trace.extracted_facts),
                    "operations_count": len(trace.memory_operations),
                    "reward": trace.reward_signal,
                    "graph_state_hash": trace.graph_state_hash[:8]
                }
                panel_data["timeline_data"].append(timeline_point)
            
            # Generate detailed action log
            for trace in traces:
                action_entry = {
                    "turn_id": trace.turn_id,
                    "input_text": trace.input_text,
                    "extracted_facts": trace.extracted_facts,
                    "memory_operations": trace.memory_operations,
                    "output_response": trace.output_response,
                    "reward_signal": trace.reward_signal,
                    "provenance_updates": len(trace.provenance_updates),
                    "graph_state_change": trace.graph_state_hash != (traces[traces.index(trace)-1].graph_state_hash if traces.index(trace) > 0 else "")
                }
                panel_data["action_log"].append(action_entry)
            
            # Calculate reward attribution
            rewards = [trace.reward_signal for trace in traces if trace.reward_signal is not None]
            operations = [op for trace in traces for op in trace.memory_operations]
            
            if rewards and operations:
                # Simple attribution: distribute rewards across operations
                operation_counts = {}
                for op in operations:
                    operation_counts[op] = operation_counts.get(op, 0) + 1
                
                total_reward = sum(rewards)
                total_operations = len(operations)
                
                panel_data["reward_attribution"] = {
                    "total_reward": total_reward,
                    "total_operations": total_operations,
                    "avg_reward_per_operation": total_reward / total_operations if total_operations > 0 else 0,
                    "operation_breakdown": {
                        op: {
                            "count": count,
                            "attributed_reward": (count / total_operations) * total_reward if total_operations > 0 else 0
                        }
                        for op, count in operation_counts.items()
                    },
                    "reward_timeline": [
                        {"turn": trace.turn_id, "reward": trace.reward_signal}
                        for trace in traces if trace.reward_signal is not None
                    ]
                }
        
        except Exception as e:
            panel_data["error"] = str(e)
        
        return panel_data
    
    def generate_qa_outcome_viewer(self) -> Dict[str, Any]:
        """Generate QA Outcome Viewer: Input question, retrieved memory, final answer, F1 score"""
        
        viewer_data = {
            "view_type": "qa_outcome",
            "timestamp": datetime.now().isoformat(),
            "qa_sessions": [],
            "performance_metrics": {},
            "memory_retrieval_analysis": {},
            "answer_quality_trends": []
        }
        
        try:
            # Get QA performance history from RL policy
            qa_history = self.memory_system.graph_rl_policy.qa_performance_history
            
            # Simulate QA sessions (in real implementation, this would come from actual QA system)
            recent_traces = list(self.memory_system.trace_buffer.traces)[-10:]  # Last 10 traces
            
            for i, trace in enumerate(recent_traces):
                # Simulate QA session based on trace
                qa_session = {
                    "session_id": f"qa_{trace.turn_id}",
                    "input_question": trace.input_text,
                    "retrieved_memory": {
                        "facts_used": trace.extracted_facts,
                        "memory_operations": trace.memory_operations,
                        "confidence_scores": [0.7 + 0.2 * np.random.random() for _ in trace.extracted_facts]  # Simulated
                    },
                    "final_answer": trace.output_response,
                    "performance_scores": {
                        "f1_score": 0.6 + 0.3 * np.random.random(),  # Simulated F1 score
                        "precision": 0.7 + 0.2 * np.random.random(),
                        "recall": 0.6 + 0.3 * np.random.random(),
                        "confidence": trace.reward_signal if trace.reward_signal else 0.5
                    },
                    "memory_effectiveness": {
                        "facts_retrieved": len(trace.extracted_facts),
                        "operations_performed": len(trace.memory_operations),
                        "retrieval_efficiency": len(trace.extracted_facts) / max(1, len(trace.memory_operations))
                    }
                }
                
                viewer_data["qa_sessions"].append(qa_session)
            
            # Calculate performance metrics
            if viewer_data["qa_sessions"]:
                f1_scores = [session["performance_scores"]["f1_score"] for session in viewer_data["qa_sessions"]]
                precision_scores = [session["performance_scores"]["precision"] for session in viewer_data["qa_sessions"]]
                recall_scores = [session["performance_scores"]["recall"] for session in viewer_data["qa_sessions"]]
                
                viewer_data["performance_metrics"] = {
                    "avg_f1_score": np.mean(f1_scores),
                    "avg_precision": np.mean(precision_scores),
                    "avg_recall": np.mean(recall_scores),
                    "f1_variance": np.var(f1_scores),
                    "total_sessions": len(viewer_data["qa_sessions"]),
                    "high_performance_sessions": sum(1 for f1 in f1_scores if f1 > 0.8)
                }
                
                # Memory retrieval analysis
                total_facts = sum(session["memory_effectiveness"]["facts_retrieved"] for session in viewer_data["qa_sessions"])
                total_operations = sum(session["memory_effectiveness"]["operations_performed"] for session in viewer_data["qa_sessions"])
                
                viewer_data["memory_retrieval_analysis"] = {
                    "total_facts_retrieved": total_facts,
                    "total_operations": total_operations,
                    "avg_facts_per_session": total_facts / len(viewer_data["qa_sessions"]),
                    "avg_operations_per_session": total_operations / len(viewer_data["qa_sessions"]),
                    "retrieval_efficiency": total_facts / max(1, total_operations)
                }
                
                # Answer quality trends
                for i, session in enumerate(viewer_data["qa_sessions"]):
                    trend_point = {
                        "session_index": i,
                        "f1_score": session["performance_scores"]["f1_score"],
                        "memory_facts": session["memory_effectiveness"]["facts_retrieved"],
                        "timestamp": datetime.now().isoformat()  # Simplified
                    }
                    viewer_data["answer_quality_trends"].append(trend_point)
        
        except Exception as e:
            viewer_data["error"] = str(e)
        
        return viewer_data
    
    def generate_policy_metrics(self) -> Dict[str, Any]:
        """Generate Policy Metrics: Reward curves, entropy, KL divergence, update stats"""
        
        metrics_data = {
            "view_type": "policy_metrics",
            "timestamp": datetime.now().isoformat(),
            "training_progress": {},
            "reward_curves": [],
            "entropy_timeline": [],
            "kl_divergence_history": [],
            "policy_weights_evolution": {},
            "performance_indicators": {}
        }
        
        try:
            rl_policy = self.memory_system.graph_rl_policy
            
            # Training progress overview
            metrics_data["training_progress"] = {
                "episodes_trained": rl_policy.training_metrics["episodes_trained"],
                "total_reward": rl_policy.training_metrics["total_reward"],
                "avg_qa_accuracy": rl_policy.training_metrics["avg_qa_accuracy"],
                "avg_graph_integrity": rl_policy.training_metrics["avg_graph_integrity"],
                "current_entropy": rl_policy.training_metrics["policy_entropy"],
                "current_kl_divergence": rl_policy.training_metrics["kl_divergence"]
            }
            
            # Reward curves from training history
            for i, episode in enumerate(rl_policy.training_history):
                reward_point = {
                    "episode": i,
                    "total_reward": episode["total_reward"],
                    "episode_length": episode["episode_length"],
                    "avg_advantage": episode["avg_advantage"],
                    "timestamp": episode["timestamp"]
                }
                metrics_data["reward_curves"].append(reward_point)
            
            # Entropy timeline
            for i, episode in enumerate(rl_policy.training_history):
                if "training_metrics" in episode and "entropy" in episode["training_metrics"]:
                    entropy_point = {
                        "episode": i,
                        "entropy": episode["training_metrics"]["entropy"],
                        "timestamp": episode["timestamp"]
                    }
                    metrics_data["entropy_timeline"].append(entropy_point)
            
            # KL divergence history
            for i, episode in enumerate(rl_policy.training_history):
                if "training_metrics" in episode and "kl_divergence" in episode["training_metrics"]:
                    kl_point = {
                        "episode": i,
                        "kl_divergence": episode["training_metrics"]["kl_divergence"],
                        "timestamp": episode["timestamp"]
                    }
                    metrics_data["kl_divergence_history"].append(kl_point)
            
            # Policy weights evolution (simplified - track key weights over time)
            key_weights = ["add_node_weight", "merge_edge_weight", "confidence_threshold"]
            
            for weight_name in key_weights:
                if weight_name in rl_policy.policy_weights:
                    # In a real implementation, you'd track weight history over time
                    metrics_data["policy_weights_evolution"][weight_name] = {
                        "current_value": rl_policy.policy_weights[weight_name],
                        "history": [rl_policy.policy_weights[weight_name]]  # Simplified
                    }
            
            # Performance indicators
            if rl_policy.training_history:
                recent_episodes = rl_policy.training_history[-5:]  # Last 5 episodes
                recent_rewards = [ep["total_reward"] for ep in recent_episodes]
                
                metrics_data["performance_indicators"] = {
                    "recent_avg_reward": np.mean(recent_rewards) if recent_rewards else 0,
                    "reward_trend": "improving" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "stable",
                    "training_stability": 1.0 - (np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8)) if recent_rewards else 0,
                    "convergence_indicator": min(1.0, rl_policy.training_metrics["episodes_trained"] / 100.0),  # Assume convergence after 100 episodes
                    "exploration_rate": rl_policy.training_metrics["policy_entropy"]
                }
        
        except Exception as e:
            metrics_data["error"] = str(e)
        
        return metrics_data
    
    def generate_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Generate complete dashboard with all sections"""
        
        dashboard = {
            "dashboard_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "system_status": self.memory_system.get_system_status(),
            "sections": {
                "graph_memory_view": self.generate_graph_memory_view(),
                "provenance_explorer": self.generate_provenance_explorer(),
                "trace_replay_panel": self.generate_trace_replay_panel(),
                "qa_outcome_viewer": self.generate_qa_outcome_viewer(),
                "policy_metrics": self.generate_policy_metrics()
            },
            "dashboard_state": dict(self.dashboard_state),
            "navigation": {
                "available_views": ["graph_memory", "provenance", "trace_replay", "qa_outcomes", "policy_metrics"],
                "current_view": self.dashboard_state["current_view"]
            }
        }
        
        return dashboard
    
    # Helper methods
    
    def _calculate_node_position(self, node_id: str, graph: nx.MultiDiGraph) -> Dict[str, float]:
        """Calculate node position for visualization"""
        # Simple hash-based positioning (in real implementation, use proper layout algorithm)
        hash_val = hash(node_id)
        x = (hash_val % 1000) / 1000.0
        y = ((hash_val // 1000) % 1000) / 1000.0
        
        return {"x": x, "y": y}
    
    def _calculate_cluster_density(self, community: set, graph: nx.MultiDiGraph) -> float:
        """Calculate density of a community/cluster"""
        subgraph = graph.subgraph(community)
        return nx.density(subgraph) if len(community) > 1 else 0.0
    
    def _generate_graph_visualization(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """Generate visualization data for graph"""
        
        # Create a simple force-directed layout simulation
        visualization = {
            "layout_type": "force_directed",
            "node_positions": {},
            "edge_paths": [],
            "color_scheme": {
                "nodes": {"entity": "#3498db", "concept": "#e74c3c", "unknown": "#95a5a6"},
                "edges": {"is_a": "#2ecc71", "has": "#f39c12", "located_in": "#9b59b6", "unknown": "#7f8c8d"}
            },
            "size_mapping": {
                "min_node_size": 5,
                "max_node_size": 20,
                "edge_width_factor": 2
            }
        }
        
        # Calculate node positions (simplified)
        for node in nodes:
            visualization["node_positions"][node["id"]] = {
                "x": node["position"]["x"] * 800,  # Scale to canvas size
                "y": node["position"]["y"] * 600,
                "size": max(5, min(20, node["degree"] * 3)),
                "color": visualization["color_scheme"]["nodes"].get(node["type"], "#95a5a6")
            }
        
        # Calculate edge paths
        for edge in edges:
            if edge["source"] in visualization["node_positions"] and edge["target"] in visualization["node_positions"]:
                source_pos = visualization["node_positions"][edge["source"]]
                target_pos = visualization["node_positions"][edge["target"]]
                
                edge_path = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "path": f"M{source_pos['x']},{source_pos['y']} L{target_pos['x']},{target_pos['y']}",
                    "color": visualization["color_scheme"]["edges"].get(edge["relation"], "#7f8c8d"),
                    "width": max(1, edge["confidence"] * 3),
                    "label": edge["relation"]
                }
                
                visualization["edge_paths"].append(edge_path)
        
        return visualization
    
    def _count_validation_statuses(self, provenance_records: Dict) -> Dict[str, int]:
        """Count validation statuses in provenance records"""
        
        status_counts = {}
        for provenance in provenance_records.values():
            status = provenance.validation_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return status_counts
    
    def update_dashboard_state(self, updates: Dict[str, Any]):
        """Update dashboard state with new parameters"""
        
        for key, value in updates.items():
            if key in self.dashboard_state:
                self.dashboard_state[key] = value
        
        print(f"📊 Dashboard state updated: {updates}")
    
    def export_dashboard_data(self, output_path: str = None) -> str:
        """Export dashboard data to JSON file"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"memory_r1_dashboard_{timestamp}.json"
        
        dashboard_data = self.generate_comprehensive_dashboard()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        print(f"📄 Dashboard data exported to {output_path}")
        return output_path

# Demo function for dashboard
def demo_memory_r1_dashboard():
    """Demonstrate the Memory-R1 dashboard functionality"""
    
    print("📊 Memory-R1 Dashboard Demo")
    print("=" * 50)
    
    # Import and initialize the memory system
    from memory_r1_modular import MemoryR1Enhanced
    
    # Create system with some sample data
    system = MemoryR1Enhanced()
    
    # Add some sample data
    sample_inputs = [
        ("Paris is the capital of France.", 0.8),
        ("France is located in Europe.", 0.9),
        ("The Eiffel Tower is in Paris.", 0.7),
        ("Europe contains many countries.", 0.6)
    ]
    
    print("📝 Adding sample data to system...")
    for input_text, reward in sample_inputs:
        system.process_input(input_text, reward)
    
    # Initialize dashboard
    dashboard = MemoryR1Dashboard(system)
    
    # Generate all dashboard sections
    print("\n📊 Generating dashboard sections...")
    
    # Graph Memory View
    graph_view = dashboard.generate_graph_memory_view()
    print(f"✅ Graph Memory View: {graph_view['graph_structure']['statistics']['total_nodes']} nodes, {graph_view['graph_structure']['statistics']['total_edges']} edges")
    
    # Provenance Explorer
    provenance_view = dashboard.generate_provenance_explorer()
    print(f"✅ Provenance Explorer: {len(provenance_view['provenance_records'])} records")
    
    # Trace Replay Panel
    trace_view = dashboard.generate_trace_replay_panel()
    print(f"✅ Trace Replay Panel: {len(trace_view['timeline_data'])} timeline points")
    
    # QA Outcome Viewer
    qa_view = dashboard.generate_qa_outcome_viewer()
    print(f"✅ QA Outcome Viewer: {len(qa_view['qa_sessions'])} QA sessions")
    
    # Policy Metrics
    policy_view = dashboard.generate_policy_metrics()
    print(f"✅ Policy Metrics: {policy_view['training_progress']['episodes_trained']} episodes trained")
    
    # Generate comprehensive dashboard
    print("\n📋 Generating comprehensive dashboard...")
    full_dashboard = dashboard.generate_comprehensive_dashboard()
    
    print(f"✅ Dashboard generated successfully!")
    print(f"   Dashboard ID: {full_dashboard['dashboard_id']}")
    print(f"   Sections: {len(full_dashboard['sections'])}")
    print(f"   System Status: {full_dashboard['system_status']['validation_status']}")
    
    # Export dashboard data
    export_path = dashboard.export_dashboard_data()
    
    print(f"\n🎉 Dashboard demo complete!")
    print(f"📄 Dashboard data exported to: {export_path}")
    
    return dashboard, full_dashboard

if __name__ == "__main__":
    demo_dashboard, dashboard_data = demo_memory_r1_dashboard()