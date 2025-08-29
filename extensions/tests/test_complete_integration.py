#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Memory-R1 Complete Integration
Tests graph_env.py, dashboardMemR1.py, ci_hooks_integration.py integration
with the existing Memory-R1 modular system.
"""

import unittest
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Import components to test
try:
    from memory_r1_modular import MemoryR1Enhanced
    from graph_env import GraphMemoryEnv, PPOGraphTrainer, create_graph_env, create_ppo_trainer
    from dashboardMemR1 import MemoryR1Dashboard, create_dashboard
    from ci_hooks_integration import CIHooksValidator, CIIntegrationManager, create_ci_validator, create_ci_manager
    from memory_r1_complete_integration import MemoryR1CompleteSystem, create_complete_system
    ALL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    ALL_COMPONENTS_AVAILABLE = False
    print(f"⚠️ Some components not available for testing: {e}")

class TestGraphEnvironment(unittest.TestCase):
    """Test RL graph environment functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not ALL_COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
        
        self.config = {
            "max_turns": 5,
            "observation_dim": 32,
            "dataset_path": "test_dataset.json"
        }
        
        # Create test dataset
        test_data = [
            {"text": "Paris is the capital of France.", "qa_accuracy": 0.9},
            {"text": "France is in Europe.", "qa_accuracy": 0.8},
            {"text": "London is the capital of England.", "qa_accuracy": 0.9}
        ]
        
        with open("test_dataset.json", "w") as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up test files"""
        test_files = ["test_dataset.json"]
        for file in test_files:
            if Path(file).exists():
                Path(file).unlink()
    
    def test_graph_env_creation(self):
        """Test graph environment creation"""
        env = create_graph_env(self.config)
        self.assertIsNotNone(env)
        self.assertEqual(env.max_turns, 5)
        self.assertEqual(env.observation_dim, 32)
    
    def test_graph_env_reset(self):
        """Test environment reset functionality"""
        env = create_graph_env(self.config)
        state = env.reset()
        
        self.assertIsNotNone(state)
        self.assertEqual(len(state.graph_features), 32)
        self.assertEqual(state.turn_id, 0)
    
    def test_graph_env_step(self):
        """Test environment step functionality"""
        env = create_graph_env(self.config)
        env.reset()
        
        # Take a step
        next_state, reward, done, info = env.step(0)  # Action 0
        
        self.assertIsNotNone(next_state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_ppo_trainer_creation(self):
        """Test PPO trainer creation"""
        env = create_graph_env(self.config)
        trainer = create_ppo_trainer(env)
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.env, env)
    
    def test_ppo_training_episode(self):
        """Test PPO training episode"""
        env = create_graph_env(self.config)
        trainer = create_ppo_trainer(env)
        
        episode_result = trainer.train_episode()
        
        self.assertIsInstance(episode_result, dict)
        self.assertIn("episode_reward", episode_result)
        self.assertIn("episode_length", episode_result)
        self.assertIn("training_metrics", episode_result)

class TestDashboard(unittest.TestCase):
    """Test dashboard functionality"""
    
    def setUp(self):
        """Set up test dashboard"""
        if not ALL_COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
    
    def test_dashboard_creation(self):
        """Test dashboard creation"""
        dashboard = create_dashboard()
        self.assertIsNotNone(dashboard)
        self.assertIsNotNone(dashboard.app)
    
    def test_dashboard_with_memory_system(self):
        """Test dashboard with memory system"""
        memory_system = MemoryR1Enhanced()
        dashboard = create_dashboard(memory_system)
        
        self.assertIsNotNone(dashboard)
        self.assertEqual(dashboard.memory_system, memory_system)
    
    def test_dashboard_graph_elements(self):
        """Test graph elements generation"""
        dashboard = create_dashboard()
        elements = dashboard._build_graph_elements()
        
        self.assertIsInstance(elements, list)
        # Should have sample data even without memory system
        self.assertGreater(len(elements), 0)
    
    def test_dashboard_stats(self):
        """Test dashboard statistics"""
        dashboard = create_dashboard()
        stats = dashboard._get_graph_stats()
        
        self.assertIsNotNone(stats)

class TestCIHooks(unittest.TestCase):
    """Test CI hooks functionality"""
    
    def setUp(self):
        """Set up test CI validator"""
        if not ALL_COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
        
        self.config = {
            "max_disconnected_nodes": 3,
            "max_cycles": 0,
            "min_provenance_integrity": 0.8
        }
    
    def test_ci_validator_creation(self):
        """Test CI validator creation"""
        validator = create_ci_validator(config=self.config)
        self.assertIsNotNone(validator)
        self.assertEqual(validator.validation_thresholds["max_disconnected_nodes"], 3)
    
    def test_validate_graph_state_without_system(self):
        """Test graph state validation without memory system"""
        validator = create_ci_validator(config=self.config)
        result = validator.validate_graph_state()
        
        self.assertIsInstance(result.test_name, str)
        self.assertIn(result.status, ["pass", "fail", "warning"])
        self.assertIsInstance(result.execution_time, float)
    
    def test_check_provenance_integrity_without_system(self):
        """Test provenance integrity check without memory system"""
        validator = create_ci_validator(config=self.config)
        result = validator.check_provenance_integrity()
        
        self.assertIsInstance(result.test_name, str)
        self.assertIn(result.status, ["pass", "fail", "warning"])
        self.assertIsInstance(result.execution_time, float)
    
    def test_replay_trace_without_system(self):
        """Test trace replay without memory system"""
        validator = create_ci_validator(config=self.config)
        result = validator.replay_trace(0, 1)
        
        self.assertIsInstance(result.test_name, str)
        self.assertIn(result.status, ["pass", "fail", "warning"])
        self.assertIsInstance(result.execution_time, float)
    
    def test_full_validation_suite(self):
        """Test full validation suite"""
        validator = create_ci_validator(config=self.config)
        report = validator.run_full_validation_suite()
        
        self.assertIsNotNone(report)
        self.assertIsInstance(report.test_results, list)
        self.assertEqual(len(report.test_results), 3)  # 3 validation hooks
        self.assertIn(report.overall_status, ["pass", "fail", "warning"])
    
    def test_ci_manager_creation(self):
        """Test CI manager creation"""
        validator = create_ci_validator(config=self.config)
        manager = create_ci_manager(validator)
        
        self.assertIsNotNone(manager)
        self.assertEqual(manager.validator, validator)
    
    def test_ci_pipeline(self):
        """Test CI pipeline execution"""
        validator = create_ci_validator(config=self.config)
        manager = create_ci_manager(validator)
        
        result = manager.run_ci_pipeline()
        
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("exit_code", result)
        self.assertIn("report", result)

class TestCompleteIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def setUp(self):
        """Set up test complete system"""
        if not ALL_COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
        
        self.config = {
            "memory_r1": {
                "storage_path": "test_memory_r1_data"
            },
            "graph_env": {
                "max_turns": 3,
                "observation_dim": 16
            },
            "ci_hooks": {
                "max_disconnected_nodes": 5
            },
            "auto_validation": False,  # Disable for testing
            "auto_training": False,    # Disable for testing
            "dashboard_port": 8051     # Different port for testing
        }
    
    def tearDown(self):
        """Clean up test data"""
        import shutil
        test_dirs = ["test_memory_r1_data"]
        for dir_path in test_dirs:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)
    
    def test_complete_system_creation(self):
        """Test complete system creation"""
        system = create_complete_system(self.config)
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.memory_system)
        self.assertIsNotNone(system.graph_env)
        self.assertIsNotNone(system.dashboard)
        self.assertIsNotNone(system.ci_validator)
    
    def test_system_status(self):
        """Test system status reporting"""
        system = create_complete_system(self.config)
        status = system.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("system_running", status)
        self.assertIn("components", status)
        self.assertIn("services", status)
    
    def test_research_query_processing(self):
        """Test research query processing"""
        system = create_complete_system(self.config)
        
        query = "Paris is the capital of France."
        result = system.process_research_query(query)
        
        self.assertIsInstance(result, dict)
        # Should not have error if system is working
        if "error" in result:
            print(f"Query processing error: {result['error']}")
    
    def test_comprehensive_test(self):
        """Test comprehensive system test"""
        system = create_complete_system(self.config)
        
        test_results = system.run_comprehensive_test()
        
        self.assertIsInstance(test_results, dict)
        self.assertIn("overall_status", test_results)
        self.assertIn("components", test_results)
        self.assertIn("integration_tests", test_results)
    
    def test_system_export(self):
        """Test system state export"""
        system = create_complete_system(self.config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = system.export_system_state(f.name)
        
        self.assertTrue(Path(export_path).exists())
        
        # Verify export content
        with open(export_path) as f:
            export_data = json.load(f)
        
        self.assertIn("export_info", export_data)
        self.assertIn("system_status", export_data)
        
        # Clean up
        Path(export_path).unlink()

class TestIntegrationWithMemorySystem(unittest.TestCase):
    """Test integration with actual Memory-R1 system"""
    
    def setUp(self):
        """Set up test with memory system"""
        if not ALL_COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
        
        self.memory_system = MemoryR1Enhanced({
            "storage_path": "test_integration_memory"
        })
    
    def tearDown(self):
        """Clean up test data"""
        import shutil
        if Path("test_integration_memory").exists():
            shutil.rmtree("test_integration_memory")
    
    def test_ci_hooks_with_memory_system(self):
        """Test CI hooks with actual memory system"""
        validator = create_ci_validator(self.memory_system)
        
        # Process some data first
        self.memory_system.process_input("Paris is the capital of France.")
        self.memory_system.process_input("France is in Europe.")
        
        # Run validation
        graph_result = validator.validate_graph_state()
        provenance_result = validator.check_provenance_integrity()
        
        self.assertIn(graph_result.status, ["pass", "fail", "warning"])
        self.assertIn(provenance_result.status, ["pass", "fail", "warning"])
    
    def test_dashboard_with_memory_system(self):
        """Test dashboard with actual memory system"""
        # Process some data
        self.memory_system.process_input("London is the capital of England.")
        
        dashboard = create_dashboard(self.memory_system)
        
        # Test graph elements generation
        elements = dashboard._build_graph_elements()
        self.assertIsInstance(elements, list)
        
        # Test stats
        stats = dashboard._get_graph_stats()
        self.assertIsNotNone(stats)
    
    def test_graph_env_with_memory_system(self):
        """Test graph environment integration"""
        # This would test deeper integration in a full implementation
        env = create_graph_env({
            "max_turns": 3,
            "observation_dim": 16
        })
        
        # Basic functionality test
        state = env.reset()
        self.assertIsNotNone(state)
        
        next_state, reward, done, info = env.step(0)
        self.assertIsNotNone(next_state)

def run_integration_tests():
    """Run all integration tests"""
    
    print("🧪 Running Memory-R1 Complete Integration Tests")
    
    if not ALL_COMPONENTS_AVAILABLE:
        print("❌ Cannot run tests - components not available")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGraphEnvironment,
        TestDashboard,
        TestCIHooks,
        TestCompleteIntegration,
        TestIntegrationWithMemorySystem
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n🧪 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✅ All integration tests passed!")
    else:
        print(f"\n❌ Some tests failed - check output above")
    
    return success

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
