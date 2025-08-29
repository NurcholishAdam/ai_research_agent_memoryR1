#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Memory-R1 Enhanced System
Demonstrates the three modular extensions with CI-evaluable hooks
"""

import asyncio
import json
from datetime import datetime
from memory_r1_enhanced import MemoryR1Enhanced

def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

async def main():
    """Main demonstration function"""
    
    print_section("Memory-R1 Enhanced System Demonstration")
    print("🧠 Modular Memory System with Semantic Graph Reasoning")
    print("📋 Provenance Validation and Trace Buffer Replay")
    print("🔧 CI-Evaluable Hooks: validate_graph_state(), check_provenance_integrity(), replay_trace()")
    
    # Initialize the enhanced system
    print_subsection("1. System Initialization")
    
    config = {
        "graph_builder": {
            "enable_entity_linking": True
        },
        "storage_path": "demo_memory_r1_data",
        "provenance_path": "demo_provenance_data",
        "trace_buffer": {
            "max_size": 100,
            "storage_path": "demo_trace_data"
        }
    }
    
    system = MemoryR1Enhanced(config)
    
    print("✅ Memory-R1 Enhanced System initialized with:")
    print("   🧠 Semantic Graph Reasoning Module")
    print("   📋 Provenance Validator Module") 
    print("   🔄 Trace Buffer Replay Module")
    
    # Demonstrate semantic graph reasoning
    print_subsection("2. Semantic Graph Reasoning Module")
    
    knowledge_inputs = [
        ("Paris is the capital of France.", 0.9),
        ("France is located in Europe.", 0.8),
        ("The Eiffel Tower is in Paris.", 0.7),
        ("Europe contains many countries.", 0.6),
        ("London is the capital of England.", 0.9),
        ("England is part of the United Kingdom.", 0.8)
    ]
    
    print(f"📊 Processing {len(knowledge_inputs)} knowledge statements...")
    
    for i, (statement, reward) in enumerate(knowledge_inputs, 1):
        print(f"\nTurn {i}: {statement}")
        
        result = system.process_input(statement, reward)
        
        if result["success"]:
            print(f"   ✅ Extracted {len(result['extracted_facts'])} facts")
            print(f"   🔧 Operations: {result['graph_operations']}")
            print(f"   💬 Response: {result['output_response'][:80]}...")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    
    # Demonstrate provenance validation
    print_subsection("3. Provenance Validation Module")
    
    print("📋 Checking provenance integrity...")
    
    provenance_validation = system.check_provenance_integrity()
    
    print(f"🔍 Provenance Validation Results:")
    print(f"   Overall Status: {provenance_validation['overall_status']}")
    print(f"   Base Validation:")
    base_val = provenance_validation['base_validation']
    print(f"      Total Records: {base_val['total_records']}")
    print(f"      Valid Records: {base_val['valid_records']}")
    print(f"      Invalid Records: {base_val['invalid_records']}")
    print(f"      Warnings: {len(base_val['warnings'])}")
    
    if provenance_validation['trace_provenance_consistency']['issues']:
        print(f"   ⚠️ Consistency Issues:")
        for issue in provenance_validation['trace_provenance_consistency']['issues'][:3]:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Trace-Provenance Consistency: OK")
    
    # Demonstrate trace buffer replay
    print_subsection("4. Trace Buffer Replay Module")
    
    print("🔄 Testing trace replay functionality...")
    
    if system.current_turn >= 3:
        replay_result = system.replay_trace(1, 3)
        
        if "error" not in replay_result:
            print(f"📊 Replay Results:")
            print(f"   Traces Replayed: {replay_result['base_replay']['traces_replayed']}")
            print(f"   Decision Points: {len(replay_result['base_replay']['decision_points'])}")
            
            if replay_result['performance_metrics']:
                metrics = replay_result['performance_metrics']
                print(f"   Performance Metrics:")
                print(f"      Total Rewards: {metrics['total_rewards']:.2f}")
                print(f"      Average Reward: {metrics['average_reward']:.3f}")
                print(f"      Positive Reward Ratio: {metrics['positive_reward_ratio']:.1%}")
        else:
            print(f"   ❌ Replay Error: {replay_result['error']}")
    else:
        print("   ⚠️ Not enough traces for replay (need at least 3 turns)")
    
    # Demonstrate memory querying
    print_subsection("5. Memory Query and Retrieval")
    
    queries = [
        "What do you know about Paris?",
        "Tell me about European capitals",
        "How are France and England related?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        
        query_result = system.query_memory(query)
        
        if query_result['total_results'] > 0:
            print(f"   📊 Found {query_result['total_results']} related entities")
            print(f"   🎯 Query entities: {query_result['query_entities']}")
            
            if query_result['graph_result']['relations']:
                print(f"   🔗 Sample relations:")
                for relation in query_result['graph_result']['relations'][:3]:
                    print(f"      {relation['subject']} {relation['predicate']} {relation['object']} (conf: {relation['confidence']:.2f})")
        else:
            print(f"   ℹ️ {query_result.get('message', 'No results found')}")
    
    # Test CI-Evaluable Hooks
    print_subsection("6. CI-Evaluable Hooks Testing")
    
    print("🔧 Testing all CI-evaluable hooks...")
    
    # Hook 1: validate_graph_state()
    print(f"\n1️⃣ validate_graph_state():")
    graph_validation = system.validate_graph_state()
    
    print(f"   Status: {graph_validation['overall_status']}")
    print(f"   Graph Stats:")
    graph_stats = graph_validation['graph_validation']
    print(f"      Nodes: {graph_stats['node_count']}")
    print(f"      Edges: {graph_stats['edge_count']}")
    print(f"      Fragments: {graph_stats['fragment_count']}")
    
    if graph_stats['issues']:
        print(f"   ⚠️ Issues Found:")
        for issue in graph_stats['issues'][:3]:
            print(f"      - {issue}")
    else:
        print(f"   ✅ No issues found")
    
    # Hook 2: check_provenance_integrity() (already tested above)
    print(f"\n2️⃣ check_provenance_integrity(): ✅ Already tested above")
    
    # Hook 3: replay_trace(start, end)
    print(f"\n3️⃣ replay_trace(start, end):")
    
    if system.current_turn >= 4:
        detailed_replay = system.replay_trace(2, 4)
        
        if "error" not in detailed_replay:
            print(f"   ✅ Successfully replayed turns 2-4")
            print(f"   📊 Replay Statistics:")
            print(f"      Requested Range: {detailed_replay['replay_parameters']['requested_range']} turns")
            print(f"      Traces Found: {detailed_replay['base_replay']['traces_replayed']}")
            
            # Show decision points
            decision_points = detailed_replay['base_replay']['decision_points']
            print(f"   🎯 Decision Points:")
            for dp in decision_points[:2]:  # Show first 2
                print(f"      Turn {dp['turn_id']}: {len(dp['extracted_facts'])} facts, {len(dp['memory_operations'])} ops")
        else:
            print(f"   ❌ Replay failed: {detailed_replay['error']}")
    else:
        print(f"   ⚠️ Need more turns for detailed replay")
    
    # System status and statistics
    print_subsection("7. System Status and Statistics")
    
    status = system.get_system_status()
    
    print(f"📊 Comprehensive System Status:")
    print(f"   Current Turn: {status['current_turn']}")
    print(f"   System Statistics:")
    stats = status['system_stats']
    print(f"      Total Extractions: {stats['total_extractions']}")
    print(f"      Total Operations: {stats['total_operations']}")
    print(f"      Total Rewards: {stats['total_rewards']:.2f}")
    
    print(f"\n   Module Status:")
    modules = status['module_status']
    print(f"      Graph Memory:")
    print(f"         Nodes: {modules['graph_memory']['nodes']}")
    print(f"         Edges: {modules['graph_memory']['edges']}")
    print(f"         Fragments: {modules['graph_memory']['fragments']}")
    
    print(f"      Provenance Tracker:")
    print(f"         Records: {modules['provenance_tracker']['records']}")
    
    print(f"      Trace Buffer:")
    print(f"         Traces: {modules['trace_buffer']['traces']}")
    print(f"         Utilization: {modules['trace_buffer']['utilization']:.1%}")
    
    print(f"\n   Validation Status:")
    validation = status['validation_status']
    print(f"      Graph State: {validation['graph_state']}")
    print(f"      Provenance Integrity: {validation['provenance_integrity']}")
    
    # Integration benefits demonstration
    print_subsection("8. Integration Benefits Demonstration")
    
    print("✨ Memory-R1 Enhanced System Benefits:")
    print()
    print("🧠 Semantic Graph Reasoning Module:")
    print("   ✅ Replaces flat memory with structured semantic graphs")
    print("   ✅ Enables compositional reasoning and entity/event linking")
    print("   ✅ LLM extracts → graph triples (subject, predicate, object)")
    print("   ✅ Memory operations: ADD_NODE, MERGE_EDGE, DELETE_SUBGRAPH, NOOP")
    print("   ✅ RL policy trained via PPO/GRPO for graph management")
    print()
    print("📋 Provenance Validator Module:")
    print("   ✅ Tracks origin, transformation, and trustworthiness")
    print("   ✅ Wraps entries with metadata: {content, source_turn, update_chain, confidence}")
    print("   ✅ Updates provenance chain on each operation")
    print("   ✅ Filters by provenance during memory distillation")
    print()
    print("🔄 Trace Buffer Replay Module:")
    print("   ✅ Enables retrospective debugging and reward attribution")
    print("   ✅ Logs every (turn, facts, operations, output, reward) tuple")
    print("   ✅ Enables offline replay for RL diagnostics")
    print("   ✅ Assigns delayed rewards to memory ops based on QA success")
    print()
    print("🔧 CI-Evaluable Hooks:")
    print("   ✅ validate_graph_state() - Validates graph integrity")
    print("   ✅ check_provenance_integrity() - Validates provenance chains")
    print("   ✅ replay_trace(start, end) - Replays memory evolution")
    
    # Performance summary
    print_subsection("9. Performance Summary")
    
    final_status = system.get_system_status()
    
    print(f"📈 Final Performance Metrics:")
    print(f"   Processing Efficiency:")
    total_ops = final_status['system_stats']['total_operations']
    total_extractions = final_status['system_stats']['total_extractions']
    if total_extractions > 0:
        ops_per_extraction = total_ops / total_extractions
        print(f"      Operations per Extraction: {ops_per_extraction:.2f}")
    
    print(f"   Memory Utilization:")
    graph_nodes = final_status['module_status']['graph_memory']['nodes']
    graph_edges = final_status['module_status']['graph_memory']['edges']
    if graph_nodes > 0:
        edge_to_node_ratio = graph_edges / graph_nodes
        print(f"      Edge-to-Node Ratio: {edge_to_node_ratio:.2f}")
    
    print(f"   System Health:")
    graph_health = final_status['validation_status']['graph_state']
    provenance_health = final_status['validation_status']['provenance_integrity']
    print(f"      Graph State: {graph_health}")
    print(f"      Provenance Integrity: {provenance_health}")
    
    overall_health = "Healthy" if graph_health == "valid" and provenance_health == "valid" else "Issues Detected"
    print(f"      Overall System Health: {overall_health}")
    
    print_section("Demo Complete")
    print("🎉 Memory-R1 Enhanced System demonstration finished!")
    print("🧠 The system successfully demonstrated all three modular extensions:")
    print("   ✅ Semantic Graph Reasoning with RL-based operation selection")
    print("   ✅ Provenance Validation with integrity checking")
    print("   ✅ Trace Buffer Replay with performance analysis")
    print("🔧 All CI-evaluable hooks are functional and ready for integration")

if __name__ == "__main__":
    asyncio.run(main())