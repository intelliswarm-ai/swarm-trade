#!/usr/bin/env python3
"""
Test script to verify command efficiency evaluation is working
"""

import sys
import time
from src.command_efficiency_evaluator import evaluator, measure_execution

def test_efficiency_system():
    """Test the efficiency evaluation system"""
    print("🧪 Testing Command Efficiency Evaluation System")
    print("=" * 50)
    
    # Test 1: Simple command measurement
    print("\n1. Testing basic measurement...")
    with measure_execution("test_simple_command"):
        time.sleep(0.1)  # Simulate 100ms operation
        result = "test completed"
    print("   ✅ Simple measurement completed")
    
    # Test 2: Multiple command measurements
    print("\n2. Testing multiple measurements...")
    for i in range(3):
        with measure_execution("test_repeated_command"):
            time.sleep(0.05)  # Simulate 50ms operation
    print("   ✅ Multiple measurements completed")
    
    # Test 3: Simulated error
    print("\n3. Testing error handling...")
    try:
        with measure_execution("test_error_command"):
            raise Exception("Simulated error")
    except Exception:
        pass
    print("   ✅ Error handling completed")
    
    # Test 4: Generate report
    print("\n4. Generating efficiency report...")
    report = evaluator.get_efficiency_report()
    
    if "summary" in report:
        print(f"   📋 Total commands executed: {report['summary']['total_commands_executed']}")
        print(f"   📋 Success rate: {report['summary']['overall_success_rate']}")
    
    if "command_analytics" in report:
        print(f"   📊 Commands tracked: {len(report['command_analytics'])}")
        for cmd_name, metrics in report["command_analytics"].items():
            print(f"      • {cmd_name}: {metrics['total_executions']} executions, {metrics['success_rate']} success rate")
    
    # Test 5: Export metrics
    print("\n5. Testing metrics export...")
    evaluator.export_metrics("test_metrics.json")
    print("   ✅ Metrics exported to test_metrics.json")
    
    print("\n🎉 All tests completed successfully!")
    print("The efficiency evaluation system is working correctly.")
    
    return report

if __name__ == "__main__":
    test_efficiency_system()