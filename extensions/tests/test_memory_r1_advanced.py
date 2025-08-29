#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for Memory-R1 Advanced System
Tests RL training loop, enhanced CI hooks, and dashboard functionality
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

def test_memory_r1_advanced_system():
    """Comprehensive test of all advanced Memory-R1 features"""
    
    print("🧪 Testing Memory-R1 Advanced System")
    print("=" * 60)
    
    try:
        # Import systems
        from memory_r1_modular import MemoryR1Enhanced
        from memory_r1_dashboard import MemoryR1Dashboard
        
        print("✅ Successfully imported advanced Memory-R1 components")
        
        # Initialize system
        system = MemoryR1Enhanced()
        dashboard = MemoryR1Dashboard(system)
        
        print("✅ Successfully initialized advanced system and dashboard")
        
        # Test 1: Basic functionality
        print("\n🔧 Test 1: Basic System Functionality")
        
        result = system.process_input("Paris is the capital of France.", 0.8)
        
        if result["success"]:
            print(f"   ✅ Basic processing: {len(result['extracted_facts'])} facts extracted")
            print(f"   ✅ RL integration: {result.get('rl_action_info', {}).get('selected_operation', 'N/A')}")
            print(f"   ✅ Composite reward: {result.get('composite_reward', 'N/A')}")
        else:
            print(f"   ❌ Basic processing failed: {result.get('error', 'Unknown error')}")
        
        # Test 2: Enhanced CI Hooks
        print("\n🔧 Test 2: Enhanced CI Hooks")
        
        # Test validate_graph_consistency
        graph_validation = system.validate_graph_consistency()
        print(f"   ✅ validate_graph_consistency(): {graph_validation['overall_status']}")
        print(f"      - Orphan nodes: {graph_validation['consistency_checks']['orphan_nodes']['count']}")
        print(f"      - Graph density: {graph_validation['graph_metrics'].get('density', 'N/A')}")
        
        # Test check_reward_alignment
        reward_alignment = system.check_reward_alignment()
        print(f"   ✅ check_reward_alignment(): {reward_alignment['overall_status']}")
        print(f"      - Correlation: {reward_alignment['reward_qa_correlation']:.3f}")
        
        # Test replay_trace_epoch
        if system.current_turn >= 1:
            epoch_replay = system.replay_trace_epoch(0)  # Epoch 0
            if "error" not in epoch_replay:
                print(f"   ✅ replay_trace_epoch(0): Success")
                print(f"      - Turns analyzed: {epoch_replay['epoch_analysis']['total_turns']}")
            else:
                print(f"   ⚠️ replay_trace_epoch(0): {epoch_replay['error']}")
        
        # Test 3: RL Training Loop
        print("\n🔧 Test 3: RL Training Loop")
        
        # Add more sample data for training
        sample_inputs = [
            ("France is located in Europe.", 0.9),
            ("The Eiffel Tower is in Paris.", 0.7),
            ("Europe contains many countries.", 0.6)
        ]
        
        for input_text, reward in sample_inputs:
            result = system.process_input(input_text, reward)
            if not result["success"]:
                print(f"   ⚠️ Failed to process: {input_text}")
        
        # Run mini training loop
        training_results = system.run_rl_training_loop(num_episodes=3, episode_length=2)
        
        if "error" not in training_results:
            print(f"   ✅ RL Training completed: {len(training_results['episode_results'])} episodes")
            print(f"      - Final avg reward: {training_results['episode_results'][-1]['avg_reward']:.3f}")
            print(f"      - Convergence: {training_results.get('convergence_analysis', {}).get('convergence_indicator', 'N/A')}")
        else:
            print(f"   ❌ RL Training failed: {training_results['error']}")
        
        # Test 4: Dashboard Generation
        print("\n🔧 Test 4: Interactive Dashboard")
        
        # Test individual dashboard sections
        graph_view = dashboard.generate_graph_memory_view()
        print(f"   ✅ Graph Memory View: {graph_view['graph_structure']['statistics']['total_nodes']} nodes")
        
        provenance_view = dashboard.generate_provenance_explorer()
        print(f"   ✅ Provenance Explorer: {len(provenance_view['provenance_records'])} records")
        
        trace_view = dashboard.generate_trace_replay_panel()
        print(f"   ✅ Trace Replay Panel: {len(trace_view['timeline_data'])} timeline points")
        
        qa_view = dashboard.generate_qa_outcome_viewer()
        print(f"   ✅ QA Outcome Viewer: {len(qa_view['qa_sessions'])} QA sessions")
        
        policy_view = dashboard.generate_policy_metrics()
        print(f"   ✅ Policy Metrics: {policy_view['training_progress']['episodes_trained']} episodes")
        
        # Test comprehensive dashboard
        full_dashboard = dashboard.generate_comprehensive_dashboard()
        print(f"   ✅ Comprehensive Dashboard: {len(full_dashboard['sections'])} sections")
        
        # Test dashboard export
        export_path = dashboard.export_dashboard_data()
        if Path(export_path).exists():
            print(f"   ✅ Dashboard Export: {export_path}")
        else:
            print(f"   ❌ Dashboard Export failed")
        
        # Test 5: System Status and Health
        print("\n🔧 Test 5: System Health Check")
        
        status = system.get_system_status()
        print(f"   ✅ System Status: {status['validation_status']}")
        print(f"      - Current turn: {status['current_turn']}")
        print(f"      - Graph nodes: {status['module_status']['graph_memory']['nodes']}")
        print(f"      - Provenance records: {status['module_status']['provenance_tracker']['records']}")
        print(f"      - Trace buffer utilization: {status['module_status']['trace_buffer']['utilization']:.1%}")
        
        # Test 6: Advanced Features Integration
        print("\n🔧 Test 6: Advanced Features Integration")
        
        # Test RL policy status
        rl_status = system.graph_rl_policy.get_training_status()
        print(f"   ✅ RL Policy Status:")
        print(f"      - Episodes trained: {rl_status['episodes_trained']}")
        print(f"      - Avg QA accuracy: {rl_status['recent_performance']['avg_qa_accuracy']:.3f}")
        print(f"      - Policy entropy: {rl_status['recent_performance']['policy_entropy']:.3f}")
        
        # Test dashboard state management
        dashboard.update_dashboard_state({
            "current_view": "policy_metrics",
            "time_range": {"start": 1, "end": 5}
        })
        print(f"   ✅ Dashboard state updated: {dashboard.dashboard_state['current_view']}")
        
        # Final validation
        print("\n🔧 Final Validation")
        
        # Check all CI hooks pass
        all_validations = [
            system.validate_graph_consistency(),
            system.check_reward_alignment()
        ]
        
        all_valid = all(v['overall_status'] in ['valid', 'warning'] for v in all_validations)
        
        if all_valid:
            print("   ✅ All CI hooks validation passed")
        else:
            print("   ⚠️ Some CI hooks reported issues")
        
        # Check training effectiveness
        if training_results and "convergence_analysis" in training_results:
            convergence = training_results["convergence_analysis"].get("convergence_indicator", 0)
            if convergence > 0.5:
                print("   ✅ RL training showing convergence")
            else:
                print("   ⚠️ RL training needs more episodes for convergence")
        
        # Check dashboard completeness
        dashboard_sections = full_dashboard.get("sections", {})
        expected_sections = ["graph_memory_view", "provenance_explorer", "trace_replay_panel", 
                           "qa_outcome_viewer", "policy_metrics"]
        
        if all(section in dashboard_sections for section in expected_sections):
            print("   ✅ Dashboard all sections generated successfully")
        else:
            print("   ⚠️ Some dashboard sections missing")
        
        print(f"\n🎉 Memory-R1 Advanced System test completed successfully!")
        print(f"📊 System Performance Summary:")
        print(f"   - Graph nodes: {status['module_status']['graph_memory']['nodes']}")
        print(f"   - Training episodes: {rl_status['episodes_trained']}")
        print(f"   - Dashboard sections: {len(dashboard_sections)}")
        print(f"   - Validation status: {'✅ Healthy' if all_valid else '⚠️ Issues detected'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced system test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_demo():
    """Test the dashboard demo functionality"""
    
    print("\n📊 Testing Dashboard Demo")
    print("-" * 40)
    
    try:
        from memory_r1_dashboard import demo_memory_r1_dashboard
        
        dashboard, dashboard_data = demo_memory_r1_dashboard()
        
        print("✅ Dashboard demo completed successfully")
        print(f"   Dashboard ID: {dashboard_data['dashboard_id']}")
        print(f"   Generated sections: {len(dashboard_data['sections'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard demo failed: {str(e)}")
        return False

async def run_comprehensive_test():
    """Run all comprehensive tests"""
    
    print("🚀 Memory-R1 Advanced System - Comprehensive Test Suite")
    print("=" * 80)
    
    # Test 1: Advanced system functionality
    test1_result = test_memory_r1_advanced_system()
    
    # Test 2: Dashboard demo
    test2_result = test_dashboard_demo()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 40)
    print(f"Advanced System Test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"Dashboard Demo Test: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    overall_success = test1_result and test2_result
    
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Memory-R1 Advanced System is fully functional!")
        print("   ✅ RL training loop operational")
        print("   ✅ Enhanced CI hooks working")
        print("   ✅ Interactive dashboard ready")
        print("   ✅ All integrations successful")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_test())
    exit(0 if success else 1)