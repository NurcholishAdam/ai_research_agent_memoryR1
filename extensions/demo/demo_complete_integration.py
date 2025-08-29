#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-R1 Complete Integration Demo
Demonstrates the integration of graph_env.py, dashboardMemR1.py, and ci_hooks_integration.py
with the existing Memory-R1 modular system.
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path

# Import all components
try:
    from memory_r1_modular import MemoryR1Enhanced
    from graph_env import create_graph_env, create_ppo_trainer
    from dashboardMemR1 import create_dashboard
    from ci_hooks_integration import create_ci_validator, create_ci_manager
    from memory_r1_complete_integration import create_complete_system
    ALL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    ALL_COMPONENTS_AVAILABLE = False
    print(f"⚠️ Some components not available: {e}")

def create_sample_training_data():
    """Create sample training data for RL environment"""
    
    training_data = [
        {"text": "Paris is the capital of France.", "qa_accuracy": 0.9},
        {"text": "France is located in Europe.", "qa_accuracy": 0.8},
        {"text": "The Eiffel Tower is in Paris.", "qa_accuracy": 0.85},
        {"text": "Europe is a continent.", "qa_accuracy": 0.7},
        {"text": "London is the capital of England.", "qa_accuracy": 0.9},
        {"text": "England is part of the United Kingdom.", "qa_accuracy": 0.8},
        {"text": "Shakespeare was born in England.", "qa_accuracy": 0.75},
        {"text": "The Thames River flows through London.", "qa_accuracy": 0.8},
        {"text": "Python is a programming language.", "qa_accuracy": 0.85},
        {"text": "Machine learning uses algorithms.", "qa_accuracy": 0.7},
        {"text": "Neural networks are used in AI.", "qa_accuracy": 0.8},
        {"text": "Deep learning is a subset of machine learning.", "qa_accuracy": 0.75},
        {"text": "Tokyo is the capital of Japan.", "qa_accuracy": 0.9},
        {"text": "Japan is an island nation.", "qa_accuracy": 0.8},
        {"text": "Mount Fuji is in Japan.", "qa_accuracy": 0.85}
    ]
    
    with open("demo_training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print("📄 Sample training data created: demo_training_data.json")
    return "demo_training_data.json"

def demo_individual_components():
    """Demonstrate individual component functionality"""
    
    print("\n🧩 === Individual Component Demonstrations ===")
    
    if not ALL_COMPONENTS_AVAILABLE:
        print("❌ Cannot run demos - components not available")
        return
    
    # 1. Memory-R1 Core System Demo
    print("\n🧠 1. Memory-R1 Core System Demo")
    print("-" * 40)
    
    memory_system = MemoryR1Enhanced({
        "storage_path": "demo_memory_r1_data"
    })
    
    # Process some sample inputs
    sample_inputs = [
        "Paris is the capital of France.",
        "France is located in Europe.",
        "The Eiffel Tower is in Paris.",
        "London is the capital of England."
    ]
    
    for input_text in sample_inputs:
        result = memory_system.process_input(input_text)
        print(f"   Processed: {input_text}")
        print(f"   Facts extracted: {len(result.get('extracted_facts', []))}")
    
    status = memory_system.get_system_status()
    print(f"   Memory system status: {status['system_stats']['total_extractions']} extractions")
    
    # 2. Graph Environment Demo
    print("\n🎮 2. Graph Environment Demo")
    print("-" * 40)
    
    # Create training data
    dataset_path = create_sample_training_data()
    
    env = create_graph_env({
        "max_turns": 5,
        "observation_dim": 32,
        "dataset_path": dataset_path
    })
    
    trainer = create_ppo_trainer(env, {
        "learning_rate": 3e-4
    })
    
    # Train a few episodes
    for episode in range(3):
        episode_result = trainer.train_episode()
        print(f"   Episode {episode + 1}: Reward = {episode_result['episode_reward']:.3f}")
    
    env_stats = env.get_environment_stats()
    print(f"   Environment stats: {env_stats['trace_statistics']['total_episodes']} episodes")
    
    # 3. CI Hooks Demo
    print("\n🔧 3. CI Hooks Demo")
    print("-" * 40)
    
    validator = create_ci_validator(memory_system, {
        "max_disconnected_nodes": 5,
        "min_provenance_integrity": 0.8
    })
    
    # Run individual validation hooks
    graph_result = validator.validate_graph_state()
    print(f"   validate_graph_state(): {graph_result.status} - {graph_result.message}")
    
    provenance_result = validator.check_provenance_integrity()
    print(f"   check_provenance_integrity(): {provenance_result.status} - {provenance_result.message}")
    
    replay_result = validator.replay_trace(0, 2)
    print(f"   replay_trace(0, 2): {replay_result.status} - {replay_result.message}")
    
    # Run full validation suite
    report = validator.run_full_validation_suite()
    print(f"   Full validation: {report.overall_status} ({report.summary['passed']} passed, {report.summary['failed']} failed)")
    
    # 4. CI Manager Demo
    print("\n🔗 4. CI Manager Demo")
    print("-" * 40)
    
    ci_manager = create_ci_manager(validator)
    pipeline_result = ci_manager.run_ci_pipeline()
    
    print(f"   CI Pipeline: {pipeline_result['status']}")
    print(f"   Exit code: {pipeline_result['exit_code']}")
    print(f"   Report: {pipeline_result['report_path']}")
    
    return memory_system, env, validator

def demo_complete_integration():
    """Demonstrate complete system integration"""
    
    print("\n🚀 === Complete System Integration Demo ===")
    
    if not ALL_COMPONENTS_AVAILABLE:
        print("❌ Cannot run integration demo - components not available")
        return
    
    # Create sample training data
    dataset_path = create_sample_training_data()
    
    # Configuration for complete system
    config = {
        "memory_r1": {
            "storage_path": "demo_complete_memory_r1_data"
        },
        "graph_env": {
            "max_turns": 10,
            "observation_dim": 32,
            "dataset_path": dataset_path
        },
        "ppo_trainer": {
            "learning_rate": 3e-4,
            "clip_epsilon": 0.2
        },
        "ci_hooks": {
            "max_disconnected_nodes": 5,
            "min_provenance_integrity": 0.8,
            "min_replay_success_rate": 0.7
        },
        "auto_validation": True,
        "validation_interval": 30,  # 30 seconds for demo
        "auto_training": False,     # Manual training for demo
        "dashboard_port": 8052      # Different port for demo
    }
    
    print(f"📋 Configuration: {json.dumps(config, indent=2)}")
    
    # Create complete system
    print("\n🏗️ Creating complete system...")
    system = create_complete_system(config)
    
    # Get initial system status
    print("\n📊 Initial system status:")
    status = system.get_system_status()
    for component, available in status["components"].items():
        print(f"   {component}: {'✅' if available else '❌'}")
    
    # Start system services
    print("\n🚀 Starting system services...")
    system.start_system()
    
    # Wait a moment for services to start
    time.sleep(2)
    
    # Process some research queries
    print("\n🔍 Processing research queries...")
    sample_queries = [
        "What is the capital of France?",
        "Where is France located?",
        "What famous tower is in Paris?",
        "What programming language is Python?",
        "What is machine learning?"
    ]
    
    for i, query in enumerate(sample_queries):
        print(f"\n   Query {i+1}: {query}")
        result = system.process_research_query(query)
        
        if "error" in result:
            print(f"      ❌ Error: {result['error']}")
        else:
            print(f"      ✅ Processed successfully")
            if "validation_status" in result:
                print(f"      🔧 Validation: {result['validation_status']}")
    
    # Run comprehensive system test
    print("\n🧪 Running comprehensive system test...")
    test_results = system.run_comprehensive_test()
    
    print(f"   Overall test status: {test_results['overall_status']}")
    print(f"   Component tests:")
    for component, result in test_results["components"].items():
        status_emoji = "✅" if result["status"] == "pass" else "❌" if result["status"] == "fail" else "⚠️"
        print(f"      {component}: {status_emoji} {result['status']} - {result['message']}")
    
    # Export system state
    print("\n📄 Exporting system state...")
    export_path = system.export_system_state()
    print(f"   System state exported to: {export_path}")
    
    # Show final system status
    print("\n📊 Final system status:")
    final_status = system.get_system_status()
    
    print(f"   System running: {final_status['system_running']}")
    print(f"   Dashboard: {'✅' if final_status['services']['dashboard_running'] else '❌'}")
    print(f"   Auto-validation: {'✅' if final_status['services']['auto_validation'] else '❌'}")
    
    if "memory_system_details" in final_status:
        memory_stats = final_status["memory_system_details"]["system_stats"]
        print(f"   Memory extractions: {memory_stats['total_extractions']}")
        print(f"   Memory operations: {memory_stats['total_operations']}")
    
    if "ci_status" in final_status:
        ci_status = final_status["ci_status"]
        print(f"   CI status: {ci_status['status']}")
        print(f"   Last validation: {ci_status.get('last_run', 'Never')}")
    
    print(f"\n🌐 Dashboard available at: http://localhost:{config['dashboard_port']}")
    print("   (Dashboard will continue running in background)")
    
    return system

def demo_dashboard_features():
    """Demonstrate dashboard features (without actually starting server)"""
    
    print("\n📊 === Dashboard Features Demo ===")
    
    if not ALL_COMPONENTS_AVAILABLE:
        print("❌ Cannot run dashboard demo - components not available")
        return
    
    # Create memory system with some data
    memory_system = MemoryR1Enhanced({
        "storage_path": "demo_dashboard_memory"
    })
    
    # Add some sample data
    sample_data = [
        "Paris is the capital of France.",
        "France is in Europe.",
        "London is the capital of England.",
        "England is part of the United Kingdom.",
        "Tokyo is the capital of Japan."
    ]
    
    for data in sample_data:
        memory_system.process_input(data)
    
    # Create graph environment
    dataset_path = create_sample_training_data()
    graph_env = create_graph_env({
        "max_turns": 5,
        "dataset_path": dataset_path
    })
    
    # Create dashboard
    dashboard = create_dashboard(memory_system, graph_env)
    
    # Demonstrate dashboard components (without starting server)
    print("📊 Dashboard components:")
    
    # Test graph elements
    graph_elements = dashboard._build_graph_elements()
    print(f"   Graph elements: {len(graph_elements)} nodes/edges")
    
    # Test graph stats
    stats = dashboard._get_graph_stats()
    print(f"   Graph statistics: Available")
    
    # Test provenance data
    provenance_data = dashboard._get_provenance_table_data()
    print(f"   Provenance entries: {len(provenance_data)}")
    
    # Load trace data
    dashboard._load_trace_data()
    print(f"   Trace entries: {len(dashboard.trace_data)}")
    
    print("   Dashboard tabs available:")
    print("      - Semantic Graph: Interactive graph visualization")
    print("      - Provenance Explorer: Provenance chains and heatmaps")
    print("      - Trace Replay: Agent decision replay")
    print("      - Validation & CI: Real-time validation status")
    print("      - RL Training: Training progress visualization")
    print("      - System Status: Component health monitoring")
    
    print(f"\n🌐 To start dashboard server:")
    print(f"   dashboard.run_server(debug=True, port=8050)")
    print(f"   Then navigate to: http://localhost:8050")

def demo_ci_integration():
    """Demonstrate CI/CD integration capabilities"""
    
    print("\n🔗 === CI/CD Integration Demo ===")
    
    if not ALL_COMPONENTS_AVAILABLE:
        print("❌ Cannot run CI demo - components not available")
        return
    
    # Create memory system
    memory_system = MemoryR1Enhanced({
        "storage_path": "demo_ci_memory"
    })
    
    # Process some data
    memory_system.process_input("Paris is the capital of France.")
    memory_system.process_input("France is in Europe.")
    
    # Create CI validator
    validator = create_ci_validator(memory_system, {
        "max_disconnected_nodes": 3,
        "max_cycles": 0,
        "min_provenance_integrity": 0.9,
        "min_replay_success_rate": 0.8
    })
    
    # Create CI manager
    ci_manager = create_ci_manager(validator, {
        "github_actions": True,
        "jenkins": False
    })
    
    print("🔧 Running CI pipeline...")
    
    # Run CI pipeline
    pipeline_result = ci_manager.run_ci_pipeline()
    
    print(f"   Pipeline status: {pipeline_result['status']}")
    print(f"   Exit code: {pipeline_result['exit_code']}")
    print(f"   Report path: {pipeline_result['report_path']}")
    
    # Show artifacts
    artifacts = pipeline_result['artifacts']
    print(f"   Generated artifacts:")
    for artifact_type, path in artifacts.items():
        print(f"      {artifact_type}: {path}")
    
    # Show validation report summary
    report = pipeline_result['report']
    print(f"\n📋 Validation Report Summary:")
    print(f"   Report ID: {report.report_id}")
    print(f"   Overall Status: {report.overall_status}")
    print(f"   Tests: {report.summary['total_tests']}")
    print(f"   Passed: {report.summary['passed']}")
    print(f"   Failed: {report.summary['failed']}")
    print(f"   Warnings: {report.summary['warnings']}")
    
    if report.recommendations:
        print(f"   Recommendations:")
        for rec in report.recommendations:
            print(f"      - {rec}")
    
    # Show CI integration examples
    print(f"\n🔗 CI Integration Examples:")
    print(f"   GitHub Actions: Set exit code {pipeline_result['exit_code']}")
    print(f"   Jenkins: Archive artifacts and publish reports")
    print(f"   Badge status: {report.overall_status}")
    
    return pipeline_result

def cleanup_demo_files():
    """Clean up demo files and directories"""
    
    print("\n🧹 Cleaning up demo files...")
    
    import shutil
    
    # Files to remove
    demo_files = [
        "demo_training_data.json",
        "memory_r1_validation_results.xml",
        "memory_r1_badge.json", 
        "memory_r1_validation_summary.md"
    ]
    
    # Directories to remove
    demo_dirs = [
        "demo_memory_r1_data",
        "demo_complete_memory_r1_data",
        "demo_dashboard_memory",
        "demo_ci_memory"
    ]
    
    # Remove files
    for file_path in demo_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"   Removed file: {file_path}")
    
    # Remove directories
    for dir_path in demo_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"   Removed directory: {dir_path}")
    
    # Remove generated reports and exports
    for pattern in ["ci_validation_report_*.json", "memory_r1_system_state_*.json", "graph_env_traces_*.json"]:
        for file_path in Path(".").glob(pattern):
            file_path.unlink()
            print(f"   Removed generated file: {file_path}")
    
    print("✅ Cleanup completed")

def main():
    """Main demo function"""
    
    print("🎭 Memory-R1 Complete Integration Demo")
    print("=" * 50)
    
    if not ALL_COMPONENTS_AVAILABLE:
        print("❌ Cannot run demo - some components not available")
        print("   Please ensure all dependencies are installed:")
        print("   pip install dash dash-cytoscape plotly pandas numpy networkx")
        return
    
    try:
        # Run individual component demos
        memory_system, env, validator = demo_individual_components()
        
        # Run complete integration demo
        complete_system = demo_complete_integration()
        
        # Demonstrate dashboard features
        demo_dashboard_features()
        
        # Demonstrate CI integration
        ci_result = demo_ci_integration()
        
        print("\n🎉 === Demo Summary ===")
        print("✅ Individual components demonstrated")
        print("✅ Complete system integration demonstrated")
        print("✅ Dashboard features showcased")
        print("✅ CI/CD integration demonstrated")
        
        print(f"\n📊 Key Results:")
        print(f"   Memory system: {memory_system.get_system_status()['system_stats']['total_extractions']} extractions")
        print(f"   RL environment: {env.get_environment_stats()['trace_statistics']['total_episodes']} episodes")
        print(f"   CI validation: {ci_result['status']}")
        
        print(f"\n🌐 Services Running:")
        if complete_system:
            status = complete_system.get_system_status()
            print(f"   Dashboard: {'✅' if status['services']['dashboard_running'] else '❌'} (port 8052)")
            print(f"   Auto-validation: {'✅' if status['services']['auto_validation'] else '❌'}")
        
        print(f"\n📚 Next Steps:")
        print(f"   1. Navigate to http://localhost:8052 to explore the dashboard")
        print(f"   2. Review generated CI reports and artifacts")
        print(f"   3. Examine exported system state files")
        print(f"   4. Run integration tests: python test_complete_integration.py")
        
        # Ask user if they want to keep services running
        try:
            response = input("\n🤔 Keep services running? (y/N): ").strip().lower()
            if response == 'y':
                print("🔄 Services will continue running...")
                print("   Press Ctrl+C to stop all services")
                while True:
                    time.sleep(10)
                    if complete_system:
                        status = complete_system.get_system_status()
                        print(f"   Heartbeat: {datetime.now().strftime('%H:%M:%S')} - System running: {status['system_running']}")
            else:
                print("🛑 Stopping services...")
                if complete_system:
                    complete_system.stop_system()
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping all services...")
            if complete_system:
                complete_system.stop_system()
    
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ask about cleanup
        try:
            response = input("\n🧹 Clean up demo files? (Y/n): ").strip().lower()
            if response != 'n':
                cleanup_demo_files()
        except KeyboardInterrupt:
            print("\n🧹 Cleaning up...")
            cleanup_demo_files()
    
    print("\n✅ Memory-R1 Complete Integration Demo finished!")

if __name__ == "__main__":
    main()
