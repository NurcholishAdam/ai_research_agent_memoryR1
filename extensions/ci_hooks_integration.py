#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI Hooks Integration System
Implements validate_graph_state(), check_provenance_integrity(), replay_trace() 
as CI-evaluable hooks with comprehensive testing and integration capabilities.
"""

import json
import uuid
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import networkx as nx
import numpy as np

# Import Memory-R1 components
try:
    from memory_r1_modular import MemoryR1Enhanced, GraphTriple, GraphFragment, TraceEntry
    from graph_env import GraphMemoryEnv, PPOGraphTrainer
    MEMORY_R1_AVAILABLE = True
except ImportError:
    MEMORY_R1_AVAILABLE = False
    print("⚠️ Memory-R1 system not available for CI hooks")

@dataclass
class CITestResult:
    """CI test result structure"""
    test_name: str
    status: str  # "pass", "fail", "warning"
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    test_results: List[CITestResult]
    overall_status: str
    summary: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime

class CIHooksValidator:
    """CI-evaluable hooks for Memory-R1 system validation"""
    
    def __init__(self, memory_system: Optional[Any] = None, config: Dict[str, Any] = None):
        self.memory_system = memory_system
        self.config = config or {}
        
        # CI configuration
        self.validation_thresholds = {
            "max_disconnected_nodes": self.config.get("max_disconnected_nodes", 5),
            "max_cycles": self.config.get("max_cycles", 0),
            "min_provenance_integrity": self.config.get("min_provenance_integrity", 0.9),
            "max_broken_chains": self.config.get("max_broken_chains", 2),
            "min_replay_success_rate": self.config.get("min_replay_success_rate", 0.8)
        }
        
        # Test history
        self.test_history: List[ValidationReport] = []
        
        print("🔧 CI Hooks Validator initialized")
        print(f"   Validation thresholds: {self.validation_thresholds}")
    
    def validate_graph_state(self) -> CITestResult:
        """
        CI Hook: validate_graph_state()
        Checks for disconnected nodes, cycles, and graph integrity issues
        """
        start_time = datetime.now()
        
        try:
            if not self.memory_system:
                return CITestResult(
                    test_name="validate_graph_state",
                    status="fail",
                    message="Memory system not available",
                    details={},
                    execution_time=0.0,
                    timestamp=start_time
                )
            
            # Run graph validation
            validation_result = self.memory_system.validate_graph_state()
            
            # Extract key metrics
            disconnected_nodes = validation_result.get("disconnected_nodes", 0)
            cycles_detected = validation_result.get("cycles_detected", 0)
            overall_status = validation_result.get("overall_status", "unknown")
            
            # Determine CI status
            ci_status = "pass"
            messages = []
            
            if disconnected_nodes > self.validation_thresholds["max_disconnected_nodes"]:
                ci_status = "fail"
                messages.append(f"Too many disconnected nodes: {disconnected_nodes}")
            
            if cycles_detected > self.validation_thresholds["max_cycles"]:
                ci_status = "fail"
                messages.append(f"Cycles detected: {cycles_detected}")
            
            if overall_status not in ["valid", "warning"]:
                ci_status = "fail"
                messages.append(f"Graph validation failed: {overall_status}")
            
            if overall_status == "warning" and ci_status == "pass":
                ci_status = "warning"
                messages.append("Graph validation returned warnings")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CITestResult(
                test_name="validate_graph_state",
                status=ci_status,
                message="; ".join(messages) if messages else "Graph state validation passed",
                details={
                    "disconnected_nodes": disconnected_nodes,
                    "cycles_detected": cycles_detected,
                    "overall_status": overall_status,
                    "validation_result": validation_result
                },
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="validate_graph_state",
                status="fail",
                message=f"Exception during validation: {str(e)}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def check_provenance_integrity(self) -> CITestResult:
        """
        CI Hook: check_provenance_integrity()
        Verifies source/update chains and provenance tracking consistency
        """
        start_time = datetime.now()
        
        try:
            if not self.memory_system:
                return CITestResult(
                    test_name="check_provenance_integrity",
                    status="fail",
                    message="Memory system not available",
                    details={},
                    execution_time=0.0,
                    timestamp=start_time
                )
            
            # Run provenance validation
            provenance_result = self.memory_system.check_provenance_integrity()
            
            # Extract key metrics
            broken_chains = provenance_result.get("broken_chains", 0)
            orphaned_entries = provenance_result.get("orphaned_entries", 0)
            integrity_score = provenance_result.get("integrity_score", 0.0)
            overall_status = provenance_result.get("overall_status", "unknown")
            
            # Determine CI status
            ci_status = "pass"
            messages = []
            
            if broken_chains > self.validation_thresholds["max_broken_chains"]:
                ci_status = "fail"
                messages.append(f"Too many broken chains: {broken_chains}")
            
            if integrity_score < self.validation_thresholds["min_provenance_integrity"]:
                ci_status = "fail"
                messages.append(f"Low integrity score: {integrity_score:.3f}")
            
            if overall_status not in ["valid", "warning"]:
                ci_status = "fail"
                messages.append(f"Provenance validation failed: {overall_status}")
            
            if overall_status == "warning" and ci_status == "pass":
                ci_status = "warning"
                messages.append("Provenance validation returned warnings")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CITestResult(
                test_name="check_provenance_integrity",
                status=ci_status,
                message="; ".join(messages) if messages else "Provenance integrity check passed",
                details={
                    "broken_chains": broken_chains,
                    "orphaned_entries": orphaned_entries,
                    "integrity_score": integrity_score,
                    "overall_status": overall_status,
                    "provenance_result": provenance_result
                },
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="check_provenance_integrity",
                status="fail",
                message=f"Exception during provenance check: {str(e)}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def replay_trace(self, start_epoch: int, end_epoch: int) -> CITestResult:
        """
        CI Hook: replay_trace(epoch)
        Reconstructs agent decisions and validates trace consistency
        """
        start_time = datetime.now()
        
        try:
            if not self.memory_system:
                return CITestResult(
                    test_name="replay_trace",
                    status="fail",
                    message="Memory system not available",
                    details={},
                    execution_time=0.0,
                    timestamp=start_time
                )
            
            # Run trace replay
            replay_result = self.memory_system.replay_trace(start_epoch, end_epoch)
            
            # Check for errors
            if "error" in replay_result:
                execution_time = (datetime.now() - start_time).total_seconds()
                return CITestResult(
                    test_name="replay_trace",
                    status="fail",
                    message=f"Replay failed: {replay_result['error']}",
                    details=replay_result,
                    execution_time=execution_time,
                    timestamp=start_time
                )
            
            # Extract key metrics
            base_replay = replay_result.get("base_replay", {})
            traces_replayed = base_replay.get("traces_replayed", 0)
            successful_replays = base_replay.get("successful_replays", 0)
            
            # Calculate success rate
            success_rate = successful_replays / traces_replayed if traces_replayed > 0 else 0.0
            
            # Determine CI status
            ci_status = "pass"
            messages = []
            
            if success_rate < self.validation_thresholds["min_replay_success_rate"]:
                ci_status = "fail"
                messages.append(f"Low replay success rate: {success_rate:.3f}")
            
            if traces_replayed == 0:
                ci_status = "warning"
                messages.append("No traces available for replay")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CITestResult(
                test_name="replay_trace",
                status=ci_status,
                message="; ".join(messages) if messages else f"Trace replay successful: {traces_replayed} traces",
                details={
                    "start_epoch": start_epoch,
                    "end_epoch": end_epoch,
                    "traces_replayed": traces_replayed,
                    "successful_replays": successful_replays,
                    "success_rate": success_rate,
                    "replay_result": replay_result
                },
                execution_time=execution_time,
                timestamp=start_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return CITestResult(
                test_name="replay_trace",
                status="fail",
                message=f"Exception during trace replay: {str(e)}",
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=start_time
            )
    
    def run_full_validation_suite(self) -> ValidationReport:
        """Run complete validation suite and generate report"""
        
        print("🔍 Running full CI validation suite...")
        
        test_results = []
        
        # Run all validation hooks
        graph_result = self.validate_graph_state()
        test_results.append(graph_result)
        print(f"   ✓ validate_graph_state(): {graph_result.status}")
        
        provenance_result = self.check_provenance_integrity()
        test_results.append(provenance_result)
        print(f"   ✓ check_provenance_integrity(): {provenance_result.status}")
        
        # Run trace replay if possible
        if self.memory_system and hasattr(self.memory_system, 'current_turn') and self.memory_system.current_turn >= 2:
            replay_result = self.replay_trace(0, min(2, self.memory_system.current_turn))
            test_results.append(replay_result)
            print(f"   ✓ replay_trace(0, 2): {replay_result.status}")
        else:
            # Create a placeholder result
            replay_result = CITestResult(
                test_name="replay_trace",
                status="warning",
                message="Insufficient data for trace replay",
                details={},
                execution_time=0.0,
                timestamp=datetime.now()
            )
            test_results.append(replay_result)
            print(f"   ⚠ replay_trace(): {replay_result.status}")
        
        # Determine overall status
        statuses = [result.status for result in test_results]
        if "fail" in statuses:
            overall_status = "fail"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "pass"
        
        # Generate summary
        summary = {
            "total_tests": len(test_results),
            "passed": sum(1 for r in test_results if r.status == "pass"),
            "failed": sum(1 for r in test_results if r.status == "fail"),
            "warnings": sum(1 for r in test_results if r.status == "warning"),
            "total_execution_time": sum(r.execution_time for r in test_results)
        }
        
        # Generate recommendations
        recommendations = []
        for result in test_results:
            if result.status == "fail":
                recommendations.append(f"Fix {result.test_name}: {result.message}")
            elif result.status == "warning":
                recommendations.append(f"Review {result.test_name}: {result.message}")
        
        # Create validation report
        report = ValidationReport(
            report_id=str(uuid.uuid4()),
            test_results=test_results,
            overall_status=overall_status,
            summary=summary,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
        
        # Store in history
        self.test_history.append(report)
        
        print(f"✅ Validation suite completed: {overall_status}")
        print(f"   Tests: {summary['passed']} passed, {summary['failed']} failed, {summary['warnings']} warnings")
        
        return report
    
    def export_validation_report(self, report: ValidationReport, output_path: str = None) -> str:
        """Export validation report to JSON file"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ci_validation_report_{timestamp}.json"
        
        # Convert report to JSON-serializable format
        report_data = {
            "report_id": report.report_id,
            "overall_status": report.overall_status,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "generated_at": report.generated_at.isoformat(),
            "test_results": [
                {
                    "test_name": result.test_name,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in report.test_results
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Validation report exported to {output_path}")
        return output_path
    
    def get_validation_history(self) -> List[ValidationReport]:
        """Get validation history"""
        return self.test_history.copy()
    
    def get_ci_status_summary(self) -> Dict[str, Any]:
        """Get CI status summary for integration"""
        
        if not self.test_history:
            return {
                "status": "unknown",
                "message": "No validation history available",
                "last_run": None
            }
        
        latest_report = self.test_history[-1]
        
        return {
            "status": latest_report.overall_status,
            "message": f"Latest validation: {latest_report.summary['passed']} passed, {latest_report.summary['failed']} failed",
            "last_run": latest_report.generated_at.isoformat(),
            "report_id": latest_report.report_id,
            "summary": latest_report.summary,
            "recommendations": latest_report.recommendations
        }

class CIIntegrationManager:
    """Manages CI integration with external systems"""
    
    def __init__(self, validator: CIHooksValidator, config: Dict[str, Any] = None):
        self.validator = validator
        self.config = config or {}
        
        # CI integration settings
        self.github_actions_mode = self.config.get("github_actions", False)
        self.jenkins_mode = self.config.get("jenkins", False)
        self.slack_webhook = self.config.get("slack_webhook")
        
        print("🔗 CI Integration Manager initialized")
    
    def run_ci_pipeline(self) -> Dict[str, Any]:
        """Run complete CI pipeline"""
        
        print("🚀 Running Memory-R1 CI Pipeline...")
        
        # Run validation suite
        report = self.validator.run_full_validation_suite()
        
        # Export report
        report_path = self.validator.export_validation_report(report)
        
        # Generate CI artifacts
        artifacts = self._generate_ci_artifacts(report)
        
        # Send notifications if configured
        if self.slack_webhook:
            self._send_slack_notification(report)
        
        # Set CI exit codes
        exit_code = 0 if report.overall_status == "pass" else 1
        
        pipeline_result = {
            "status": report.overall_status,
            "exit_code": exit_code,
            "report": report,
            "report_path": report_path,
            "artifacts": artifacts,
            "summary": report.summary
        }
        
        print(f"🏁 CI Pipeline completed with status: {report.overall_status}")
        
        return pipeline_result
    
    def _generate_ci_artifacts(self, report: ValidationReport) -> Dict[str, str]:
        """Generate CI artifacts"""
        
        artifacts = {}
        
        # Generate JUnit XML for test reporting
        junit_path = self._generate_junit_xml(report)
        artifacts["junit_xml"] = junit_path
        
        # Generate badge data
        badge_data = self._generate_badge_data(report)
        artifacts["badge_data"] = badge_data
        
        # Generate summary markdown
        summary_path = self._generate_summary_markdown(report)
        artifacts["summary_markdown"] = summary_path
        
        return artifacts
    
    def _generate_junit_xml(self, report: ValidationReport) -> str:
        """Generate JUnit XML for CI systems"""
        
        junit_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Memory-R1 Validation" tests="{report.summary['total_tests']}" 
           failures="{report.summary['failed']}" errors="0" 
           time="{report.summary['total_execution_time']:.3f}">
"""
        
        for result in report.test_results:
            junit_xml += f"""  <testcase name="{result.test_name}" 
                     classname="MemoryR1Validation" 
                     time="{result.execution_time:.3f}">
"""
            
            if result.status == "fail":
                junit_xml += f"""    <failure message="{result.message}">
      {json.dumps(result.details, indent=2)}
    </failure>
"""
            elif result.status == "warning":
                junit_xml += f"""    <skipped message="{result.message}"/>
"""
            
            junit_xml += "  </testcase>\n"
        
        junit_xml += "</testsuite>"
        
        junit_path = "memory_r1_validation_results.xml"
        with open(junit_path, "w", encoding="utf-8") as f:
            f.write(junit_xml)
        
        return junit_path
    
    def _generate_badge_data(self, report: ValidationReport) -> str:
        """Generate badge data for README"""
        
        status_color = {
            "pass": "brightgreen",
            "warning": "yellow", 
            "fail": "red"
        }
        
        badge_data = {
            "schemaVersion": 1,
            "label": "Memory-R1 Validation",
            "message": report.overall_status,
            "color": status_color.get(report.overall_status, "lightgrey")
        }
        
        badge_path = "memory_r1_badge.json"
        with open(badge_path, "w", encoding="utf-8") as f:
            json.dump(badge_data, f, indent=2)
        
        return badge_path
    
    def _generate_summary_markdown(self, report: ValidationReport) -> str:
        """Generate summary markdown for CI"""
        
        status_emoji = {
            "pass": "✅",
            "warning": "⚠️",
            "fail": "❌"
        }
        
        markdown = f"""# Memory-R1 Validation Report
        
## Overall Status: {status_emoji.get(report.overall_status, '❓')} {report.overall_status.upper()}

### Summary
- **Total Tests**: {report.summary['total_tests']}
- **Passed**: {report.summary['passed']}
- **Failed**: {report.summary['failed']}
- **Warnings**: {report.summary['warnings']}
- **Execution Time**: {report.summary['total_execution_time']:.3f}s

### Test Results
"""
        
        for result in report.test_results:
            emoji = status_emoji.get(result.status, '❓')
            markdown += f"- {emoji} **{result.test_name}**: {result.message}\n"
        
        if report.recommendations:
            markdown += "\n### Recommendations\n"
            for rec in report.recommendations:
                markdown += f"- {rec}\n"
        
        markdown += f"\n*Generated at: {report.generated_at.isoformat()}*\n"
        
        summary_path = "memory_r1_validation_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        return summary_path
    
    def _send_slack_notification(self, report: ValidationReport):
        """Send Slack notification (placeholder)"""
        
        # In a real implementation, this would send to Slack webhook
        print(f"📢 Slack notification: Memory-R1 validation {report.overall_status}")

# Utility functions
def create_ci_validator(memory_system=None, config=None) -> CIHooksValidator:
    """Create CI hooks validator"""
    return CIHooksValidator(memory_system, config)

def create_ci_manager(validator: CIHooksValidator, config=None) -> CIIntegrationManager:
    """Create CI integration manager"""
    return CIIntegrationManager(validator, config)

def run_ci_validation_demo():
    """Run CI validation demonstration"""
    
    print("🔧 Memory-R1 CI Validation Demo")
    
    # Create validator without memory system for demo
    validator = create_ci_validator()
    
    # Create CI manager
    ci_manager = create_ci_manager(validator)
    
    # Run CI pipeline
    result = ci_manager.run_ci_pipeline()
    
    print(f"\n✅ CI Demo completed!")
    print(f"   Status: {result['status']}")
    print(f"   Exit code: {result['exit_code']}")
    print(f"   Report: {result['report_path']}")

if __name__ == "__main__":
    run_ci_validation_demo()
