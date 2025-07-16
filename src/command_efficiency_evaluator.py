"""
Command Efficiency Evaluation Framework

This module provides comprehensive evaluation metrics for command efficiency
including execution time, resource usage, success rates, and quality metrics.
"""

import time
import psutil
import asyncio
import functools
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import statistics
from contextlib import contextmanager


@dataclass
class CommandMetrics:
    """Stores metrics for a single command execution"""
    command_name: str
    start_time: float
    end_time: float
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    result_size_bytes: int = 0
    token_usage: int = 0
    api_calls_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple command executions"""
    command_name: str
    total_executions: int
    success_rate: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    median_execution_time: float
    avg_memory_usage: float
    avg_cpu_usage: float
    total_token_usage: int
    avg_tokens_per_execution: float
    error_types: Dict[str, int]


class CommandEfficiencyEvaluator:
    """Main class for evaluating command efficiency"""
    
    def __init__(self):
        self.metrics_history: List[CommandMetrics] = []
        self.process = psutil.Process()
        
    def measure_command(self, command_name: str):
        """Decorator to measure command execution metrics"""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_metrics(command_name, func, *args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_metrics(command_name, func, *args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    async def _execute_with_metrics(self, command_name: str, func: Callable, *args, **kwargs):
        """Execute function while collecting metrics"""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Monitor CPU usage during execution
            cpu_before = self.process.cpu_percent()
            
            # Execute the command
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = self.process.cpu_percent()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = max(end_memory - start_memory, 0)
            cpu_usage = (cpu_before + cpu_after) / 2
            
            # Estimate result size
            result_size = self._estimate_size(result)
            
            # Create metrics record
            metrics = CommandMetrics(
                command_name=command_name,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_usage,
                success=True,
                result_size_bytes=result_size
            )
            
            self.metrics_history.append(metrics)
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            metrics = CommandMetrics(
                command_name=command_name,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_percent=0,
                success=False,
                error_message=str(e)
            )
            
            self.metrics_history.append(metrics)
            raise
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate the size of an object in bytes"""
        try:
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (dict, list)):
                return len(json.dumps(obj, default=str).encode('utf-8'))
            else:
                return len(str(obj).encode('utf-8'))
        except:
            return 0
    
    def get_command_analytics(self, command_name: str) -> Optional[AggregatedMetrics]:
        """Get aggregated analytics for a specific command"""
        command_metrics = [m for m in self.metrics_history if m.command_name == command_name]
        
        if not command_metrics:
            return None
        
        successful_metrics = [m for m in command_metrics if m.success]
        execution_times = [m.execution_time for m in successful_metrics]
        
        if not execution_times:
            execution_times = [0]
        
        # Collect error types
        error_types = {}
        for m in command_metrics:
            if not m.success and m.error_message:
                error_type = type(Exception(m.error_message)).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return AggregatedMetrics(
            command_name=command_name,
            total_executions=len(command_metrics),
            success_rate=len(successful_metrics) / len(command_metrics),
            avg_execution_time=statistics.mean(execution_times),
            min_execution_time=min(execution_times),
            max_execution_time=max(execution_times),
            median_execution_time=statistics.median(execution_times),
            avg_memory_usage=statistics.mean([m.memory_usage_mb for m in successful_metrics]) if successful_metrics else 0,
            avg_cpu_usage=statistics.mean([m.cpu_percent for m in successful_metrics]) if successful_metrics else 0,
            total_token_usage=sum([m.token_usage for m in command_metrics]),
            avg_tokens_per_execution=statistics.mean([m.token_usage for m in command_metrics if m.token_usage > 0]) if any(m.token_usage > 0 for m in command_metrics) else 0,
            error_types=error_types
        )
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generate comprehensive efficiency report"""
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}
        
        # Get all unique command names
        command_names = list(set(m.command_name for m in self.metrics_history))
        
        # Generate analytics for each command
        command_analytics = {}
        for cmd_name in command_names:
            analytics = self.get_command_analytics(cmd_name)
            if analytics:
                command_analytics[cmd_name] = {
                    "total_executions": analytics.total_executions,
                    "success_rate": f"{analytics.success_rate:.2%}",
                    "avg_execution_time": f"{analytics.avg_execution_time:.3f}s",
                    "min_execution_time": f"{analytics.min_execution_time:.3f}s",
                    "max_execution_time": f"{analytics.max_execution_time:.3f}s",
                    "median_execution_time": f"{analytics.median_execution_time:.3f}s",
                    "avg_memory_usage": f"{analytics.avg_memory_usage:.2f}MB",
                    "avg_cpu_usage": f"{analytics.avg_cpu_usage:.1f}%",
                    "total_token_usage": analytics.total_token_usage,
                    "avg_tokens_per_execution": f"{analytics.avg_tokens_per_execution:.1f}",
                    "error_types": analytics.error_types
                }
        
        # Overall system metrics
        total_executions = len(self.metrics_history)
        successful_executions = len([m for m in self.metrics_history if m.success])
        
        # Performance rankings
        sorted_by_speed = sorted(command_names, 
                               key=lambda x: self.get_command_analytics(x).avg_execution_time)
        sorted_by_success = sorted(command_names, 
                                 key=lambda x: self.get_command_analytics(x).success_rate, 
                                 reverse=True)
        
        return {
            "summary": {
                "total_commands_executed": total_executions,
                "overall_success_rate": f"{successful_executions/total_executions:.2%}" if total_executions > 0 else "0%",
                "total_unique_commands": len(command_names),
                "data_collection_period": f"{self.metrics_history[0].timestamp} to {self.metrics_history[-1].timestamp}"
            },
            "command_analytics": command_analytics,
            "performance_rankings": {
                "fastest_commands": sorted_by_speed[:5],
                "most_reliable_commands": sorted_by_success[:5],
                "slowest_commands": sorted_by_speed[-5:] if len(sorted_by_speed) > 5 else []
            },
            "recommendations": self._generate_recommendations(command_analytics)
        }
    
    def _generate_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        for cmd_name, metrics in analytics.items():
            success_rate = float(metrics["success_rate"].strip('%')) / 100
            avg_time = float(metrics["avg_execution_time"].strip('s'))
            
            if success_rate < 0.9:
                recommendations.append(f"Improve reliability of '{cmd_name}' command (success rate: {metrics['success_rate']})")
            
            if avg_time > 5.0:
                recommendations.append(f"Optimize '{cmd_name}' command performance (avg time: {metrics['avg_execution_time']})")
            
            if metrics["error_types"]:
                main_error = max(metrics["error_types"].items(), key=lambda x: x[1])
                recommendations.append(f"Address '{main_error[0]}' errors in '{cmd_name}' command ({main_error[1]} occurrences)")
        
        if not recommendations:
            recommendations.append("All commands are performing within acceptable parameters")
        
        return recommendations
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        export_data = {
            "metrics_history": [
                {
                    "command_name": m.command_name,
                    "execution_time": m.execution_time,
                    "memory_usage_mb": m.memory_usage_mb,
                    "cpu_percent": m.cpu_percent,
                    "success": m.success,
                    "error_message": m.error_message,
                    "result_size_bytes": m.result_size_bytes,
                    "token_usage": m.token_usage,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.metrics_history
            ],
            "efficiency_report": self.get_efficiency_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        self.metrics_history.clear()


# Global evaluator instance
evaluator = CommandEfficiencyEvaluator()


# Convenience decorators
def measure_command(command_name: str):
    """Decorator for measuring command efficiency"""
    return evaluator.measure_command(command_name)


# Context manager for manual measurement
@contextmanager
def measure_execution(command_name: str):
    """Context manager for measuring code block execution"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        raise
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics = CommandMetrics(
            command_name=command_name,
            start_time=start_time,
            end_time=end_time,
            execution_time=end_time - start_time,
            memory_usage_mb=max(end_memory - start_memory, 0),
            cpu_percent=psutil.Process().cpu_percent(),
            success=success,
            error_message=error_msg
        )
        
        evaluator.metrics_history.append(metrics)