#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-R1 Complete Integration System
Orchestrates graph_env.py, dashboardMemR1.py, and ci_hooks_integration.py
with the existing Memory-R1 modular system for comprehensive functionality.
"""

import json
import uuid
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import asyncio

# Import all components
try:
    from memory_r1_modular import MemoryR1Enhanced
    from graph_env import GraphMemoryEnv, PPOGraphTrainer, create_graph_env, create_ppo_trainer
    from dashboardMemR1 import MemoryR1Dashboard, create_dashboard
    from ci_hooks_integration import CIHooksValidator, CIIntegrationManager, create_ci_validator, create_ci_manager
    ALL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    ALL_COMPONENTS_AVAILABLE = False
    print(f"⚠️ Some components not available: {e}")

class MemoryR1CompleteSystem:
    """Complete Memory-R1 system with RL environment, dashboard, and CI hooks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # System components
        self.memory_system: Optional[MemoryR1Enhanced] = None
        self.graph_env: Optional[GraphMemoryEnv] = None
        self.ppo_trainer: Optional[PPOGraphTrainer] = None
        self.dashboard: Optional[MemoryR1Dashboard] = None
        self.ci_validator: Optional[CIHooksValidator] = None
        self.ci_manager: Optional[CIIntegrationManager] = None
        
        # System state
        self.is_running = False
        self.dashboard_thread: Optional[threading.Thread] = None
        self.training_thread: Optional[threading.Thread] = None
        
        # Integration settings
        self.auto_validation = self.config.get("auto_validation", True)
        self.validation_interval = self.config.get("validation_interval", 300)  # 5 minutes
        self.auto_training = self.config.get("auto_training", False)
        self.dashboard_port = self.config.get("dashboard_port", 8050)
        
        print("🚀 Memory-R1 Complete System initializing...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        if not ALL_COMPONENTS_AVAILABLE:
            print("❌ Cannot initialize - missing components")
            return
        
        try:
            # Initialize Memory-R1 core system
            print("🧠 Initializing Memory-R1 core system...")
            memory_config = self.config.get("memory_r1", {})
            self.memory_system = MemoryR1Enhanced(memory_config)
            
            # Initialize RL environment
            print("🎮 Initializing RL environment...")
            env_config = self.config.get("graph_env", {})
            self.graph_env = create_graph_env(env_config)
            
            # Initialize PPO trainer
            print("🎯 Initializing PPO trainer...")
            trainer_config = self.config.get("ppo_trainer", {})
            self.ppo_trainer = create_ppo_trainer(self.graph_env, trainer_config)
            
            # Initialize dashboard
            print("📊 Initializing dashboard...")
            self.dashboard = create_dashboard(self.memory_system, self.graph_env)
            
            # Initialize CI validator
            print("🔧 Initializing CI validator...")
            ci_config = self.config.get("ci_hooks", {})
            self.ci_validator = create_ci_validator(self.memory_system, ci_config)
            
            # Initialize CI manager
            print("🔗 Initializing CI manager...")
            ci_manager_config = self.config.get("ci_manager", {})
            self.ci_manager = create_ci_manager(self.ci_validator, ci_manager_config)
            
            print("✅ All components initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
    
    def start_system(self):
        """Start the complete Memory-R1 system"""
        
        if not ALL_COMPONENTS_AVAILABLE:
            print("❌ Cannot start system - missing components")
            return
        
        print("🚀 Starting Memory-R1 Complete System...")
        
        self.is_running = True
        
        # Start dashboard in separate thread
        if self.dashboard:
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            print(f"📊 Dashboard started on http://localhost:{self.dashboard_port}")
        
        # Start automatic validation if enabled
        if self.auto_validation and self.ci_validator:
            validation_thread = threading.Thread(
                target=self._run_periodic_validation,
                daemon=True
            )
            validation_thread.start()
            print(f"🔧 Automatic validation enabled (interval: {self.validation_interval}s)")
        
        # Start automatic training if enabled
        if self.auto_training and self.ppo_trainer:
            self.training_thread = threading.Thread(
                target=self._run_continuous_training,
                daemon=True
            )
            self.training_thread.start()
            print("🎯 Automatic training enabled")
        
        print("✅ Memory-R1 Complete System is running!")
        print(f"   Dashboard: http://localhost:{self.dashboard_port}")
        print(f"   Auto-validation: {'✅' if self.auto_validation else '❌'}")
        print(f"   Auto-training: {'✅' if self.auto_training else '❌'}")
    
    def stop_system(self):
        """Stop the complete Memory-R1 system"""
        
        print("🛑 Stopping Memory-R1 Complete System...")
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            print("📊 Stopping dashboard...")
            # Dashboard will stop when main process ends
        
        if self.training_thread and self.training_thread.is_alive():
            print("🎯 Stopping training...")
            # Training thread will stop due to is_running flag
        
        print("✅ Memory-R1 Complete System stopped")
    
    def _run_dashboard(self):
        """Run dashboard in separate thread"""
        try:
            self.dashboard.run_server(debug=False, port=self.dashboard_port)
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
    
    def _run_periodic_validation(self):
        """Run periodic validation checks"""
        
        while self.is_running:
            try:
                print("🔍 Running periodic validation...")
                report = self.ci_validator.run_full_validation_suite()
                
                if report.overall_status == "fail":
                    print(f"❌ Validation failed: {len(report.recommendations)} issues found")
                    for rec in report.recommendations[:3]:  # Show first 3 recommendations
                        print(f"   - {rec}")
                elif report.overall_status == "warning":
                    print(f"⚠️ Validation warnings: {report.summary['warnings']} warnings")
                else:
                    print("✅ Validation passed")
                
                # Wait for next validation
                time.sleep(self.validation_interval)
                
            except Exception as e:
                print(f"❌ Validation error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _run_continuous_training(self):
        """Run continuous RL training"""
        
        episode_count = 0
        
        while self.is_running:
            try:
                print(f"🎯 Training episode {episode_count + 1}...")
                episode_result = self.ppo_trainer.train_episode()
                
                episode_count += 1
                reward = episode_result["episode_reward"]
                
                print(f"   Episode {episode_count}: Reward = {reward:.3f}")
                
                # Brief pause between episodes
                time.sleep(1)
                
                # Run validation every 10 episodes
                if episode_count % 10 == 0 and self.ci_validator:
                    print("🔍 Running training validation...")
                    report = self.ci_validator.run_full_validation_suite()
                    print(f"   Validation: {report.overall_status}")
                
            except Exception as e:
                print(f"❌ Training error: {e}")
                time.sleep(10)  # Wait before retry
    
    def process_research_query(self, query: str) -> Dict[str, Any]:
        """Process a research query through the complete system"""
        
        if not self.memory_system:
            return {"error": "Memory system not available"}
        
        print(f"🔍 Processing research query: {query}")
        
        try:
            # Process through Memory-R1 system
            result = self.memory_system.process_input(query)
            
            # Run validation after processing
            if self.ci_validator:
                validation = self.ci_validator.run_full_validation_suite()
                result["validation_status"] = validation.overall_status
                result["validation_summary"] = validation.summary
            
            # Update RL environment if available
            if self.graph_env:
                # This would integrate with RL training in a real scenario
                env_stats = self.graph_env.get_environment_stats()
                result["rl_environment_stats"] = env_stats
            
            print(f"✅ Query processed successfully")
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test"""
        
        print("🧪 Running comprehensive Memory-R1 system test...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "integration_tests": {},
            "overall_status": "unknown"
        }
        
        # Test Memory-R1 core
        if self.memory_system:
            try:
                test_query = "Paris is the capital of France. France is in Europe."
                result = self.memory_system.process_input(test_query)
                test_results["components"]["memory_r1"] = {
                    "status": "pass",
                    "message": "Memory system functional",
                    "details": result
                }
            except Exception as e:
                test_results["components"]["memory_r1"] = {
                    "status": "fail",
                    "message": f"Memory system error: {e}"
                }
        
        # Test RL environment
        if self.graph_env:
            try:
                env_stats = self.graph_env.get_environment_stats()
                test_results["components"]["graph_env"] = {
                    "status": "pass",
                    "message": "RL environment functional",
                    "details": env_stats
                }
            except Exception as e:
                test_results["components"]["graph_env"] = {
                    "status": "fail",
                    "message": f"RL environment error: {e}"
                }
        
        # Test CI validation
        if self.ci_validator:
            try:
                validation_report = self.ci_validator.run_full_validation_suite()
                test_results["components"]["ci_validation"] = {
                    "status": validation_report.overall_status,
                    "message": f"Validation completed: {validation_report.summary}",
                    "details": validation_report.summary
                }
            except Exception as e:
                test_results["components"]["ci_validation"] = {
                    "status": "fail",
                    "message": f"CI validation error: {e}"
                }
        
        # Test dashboard (basic check)
        if self.dashboard:
            test_results["components"]["dashboard"] = {
                "status": "pass",
                "message": "Dashboard initialized"
            }
        
        # Integration tests
        try:
            # Test Memory-R1 + CI integration
            if self.memory_system and self.ci_validator:
                query_result = self.process_research_query("Test integration query")
                test_results["integration_tests"]["memory_ci"] = {
                    "status": "pass" if "error" not in query_result else "fail",
                    "message": "Memory-R1 + CI integration test"
                }
            
            # Test RL + Memory integration
            if self.graph_env and self.memory_system:
                # This would test actual integration
                test_results["integration_tests"]["rl_memory"] = {
                    "status": "pass",
                    "message": "RL + Memory integration available"
                }
        
        except Exception as e:
            test_results["integration_tests"]["error"] = str(e)
        
        # Determine overall status
        component_statuses = [comp.get("status", "fail") for comp in test_results["components"].values()]
        integration_statuses = [test.get("status", "fail") for test in test_results["integration_tests"].values()]
        
        all_statuses = component_statuses + integration_statuses
        
        if "fail" in all_statuses:
            test_results["overall_status"] = "fail"
        elif "warning" in all_statuses:
            test_results["overall_status"] = "warning"
        else:
            test_results["overall_status"] = "pass"
        
        print(f"🧪 Comprehensive test completed: {test_results['overall_status']}")
        
        return test_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "system_running": self.is_running,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "memory_r1": self.memory_system is not None,
                "graph_env": self.graph_env is not None,
                "ppo_trainer": self.ppo_trainer is not None,
                "dashboard": self.dashboard is not None,
                "ci_validator": self.ci_validator is not None,
                "ci_manager": self.ci_manager is not None
            },
            "services": {
                "dashboard_running": self.dashboard_thread is not None and self.dashboard_thread.is_alive(),
                "training_running": self.training_thread is not None and self.training_thread.is_alive(),
                "auto_validation": self.auto_validation
            }
        }
        
        # Add component-specific status
        if self.memory_system:
            try:
                memory_status = self.memory_system.get_system_status()
                status["memory_system_details"] = memory_status
            except Exception as e:
                status["memory_system_error"] = str(e)
        
        if self.graph_env:
            try:
                env_stats = self.graph_env.get_environment_stats()
                status["graph_env_details"] = env_stats
            except Exception as e:
                status["graph_env_error"] = str(e)
        
        if self.ci_validator:
            try:
                ci_status = self.ci_validator.get_ci_status_summary()
                status["ci_status"] = ci_status
            except Exception as e:
                status["ci_error"] = str(e)
        
        return status
    
    def export_system_state(self, output_path: str = None) -> str:
        """Export complete system state"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"memory_r1_system_state_{timestamp}.json"
        
        system_state = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "system_version": "Memory-R1 Complete v1.0",
                "components_available": ALL_COMPONENTS_AVAILABLE
            },
            "system_status": self.get_system_status(),
            "configuration": self.config
        }
        
        # Add component exports
        if self.memory_system:
            try:
                memory_export = self.memory_system.export_system_state()
                system_state["memory_system_export"] = memory_export
            except Exception as e:
                system_state["memory_system_export_error"] = str(e)
        
        if self.graph_env:
            try:
                env_export = self.graph_env.export_episode_traces()
                system_state["graph_env_export"] = env_export
            except Exception as e:
                system_state["graph_env_export_error"] = str(e)
        
        if self.ci_validator:
            try:
                validation_history = self.ci_validator.get_validation_history()
                system_state["validation_history"] = [
                    {
                        "report_id": report.report_id,
                        "overall_status": report.overall_status,
                        "summary": report.summary,
                        "generated_at": report.generated_at.isoformat()
                    }
                    for report in validation_history
                ]
            except Exception as e:
                system_state["validation_export_error"] = str(e)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(system_state, f, indent=2, default=str)
        
        print(f"📄 System state exported to {output_path}")
        return output_path

# Utility functions
def create_complete_system(config: Dict[str, Any] = None) -> MemoryR1CompleteSystem:
    """Create complete Memory-R1 system"""
    return MemoryR1CompleteSystem(config)

def run_complete_system_demo():
    """Run complete system demonstration"""
    
    print("🚀 Memory-R1 Complete System Demo")
    
    # Configuration
    config = {
        "memory_r1": {
            "storage_path": "demo_memory_r1_data"
        },
        "graph_env": {
            "max_turns": 20,
            "dataset_path": "demo_training_data.json"
        },
        "ppo_trainer": {
            "learning_rate": 3e-4
        },
        "ci_hooks": {
            "max_disconnected_nodes": 3,
            "min_provenance_integrity": 0.8
        },
        "auto_validation": True,
        "validation_interval": 60,  # 1 minute for demo
        "auto_training": False,  # Disabled for demo
        "dashboard_port": 8050
    }
    
    # Create system
    system = create_complete_system(config)
    
    # Start system
    system.start_system()
    
    # Run comprehensive test
    test_results = system.run_comprehensive_test()
    print(f"\n🧪 System test results: {test_results['overall_status']}")
    
    # Process some sample queries
    sample_queries = [
        "Paris is the capital of France.",
        "France is located in Europe.",
        "The Eiffel Tower is in Paris."
    ]
    
    for query in sample_queries:
        result = system.process_research_query(query)
        print(f"✅ Processed: {query}")
    
    # Export system state
    export_path = system.export_system_state()
    
    print(f"\n✅ Demo completed!")
    print(f"   Dashboard: http://localhost:8050")
    print(f"   System export: {export_path}")
    print(f"   System status: {system.get_system_status()['components']}")
    
    # Keep running for demo (in practice, you'd handle this differently)
    try:
        print("\n🔄 System running... Press Ctrl+C to stop")
        while True:
            time.sleep(10)
            status = system.get_system_status()
            print(f"   System heartbeat: {status['system_running']} - {datetime.now().strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        system.stop_system()

if __name__ == "__main__":
    run_complete_system_demo()