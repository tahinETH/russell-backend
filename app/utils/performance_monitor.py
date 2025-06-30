import time
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and log performance metrics for speech generation pipeline"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.metrics[operation] = duration
        del self.start_times[operation]
        
        logger.info(f"Operation '{operation}' took {duration:.3f} seconds")
        return duration
    
    @asynccontextmanager
    async def measure(self, operation: str):
        """Async context manager for measuring operation time"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.metrics[operation] = duration
            logger.info(f"Operation '{operation}' took {duration:.3f} seconds")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all collected metrics"""
        return self.metrics.copy()
    
    def log_summary(self) -> None:
        """Log a summary of all metrics"""
        if not self.metrics:
            logger.info("No metrics collected")
            return
        
        total_time = sum(self.metrics.values())
        logger.info("=== Performance Summary ===")
        logger.info(f"Total time: {total_time:.3f} seconds")
        
        # Sort by duration (longest first)
        sorted_metrics = sorted(
            self.metrics.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for operation, duration in sorted_metrics:
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            logger.info(f"{operation}: {duration:.3f}s ({percentage:.1f}%)")
    
    def identify_bottlenecks(self, threshold_percentage: float = 30.0) -> list:
        """Identify operations that take more than threshold percentage of total time"""
        if not self.metrics:
            return []
        
        total_time = sum(self.metrics.values())
        bottlenecks = []
        
        for operation, duration in self.metrics.items():
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            if percentage >= threshold_percentage:
                bottlenecks.append({
                    "operation": operation,
                    "duration": duration,
                    "percentage": percentage
                })
        
        return sorted(bottlenecks, key=lambda x: x["percentage"], reverse=True)


class StreamingMetrics:
    """Track metrics for streaming operations"""
    
    def __init__(self):
        self.first_chunk_time = None
        self.last_chunk_time = None
        self.chunk_count = 0
        self.total_bytes = 0
        self.start_time = time.time()
    
    def record_chunk(self, chunk_size: int) -> None:
        """Record a chunk being processed"""
        current_time = time.time()
        
        if self.first_chunk_time is None:
            self.first_chunk_time = current_time
            logger.debug(f"First chunk received after {current_time - self.start_time:.3f}s")
        
        self.last_chunk_time = current_time
        self.chunk_count += 1
        self.total_bytes += chunk_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        if self.first_chunk_time is None:
            return {
                "status": "no_chunks_received",
                "total_time": time.time() - self.start_time
            }
        
        total_time = self.last_chunk_time - self.start_time
        streaming_time = self.last_chunk_time - self.first_chunk_time
        time_to_first_chunk = self.first_chunk_time - self.start_time
        
        return {
            "total_time": total_time,
            "time_to_first_chunk": time_to_first_chunk,
            "streaming_time": streaming_time,
            "chunk_count": self.chunk_count,
            "total_bytes": self.total_bytes,
            "average_chunk_size": self.total_bytes / self.chunk_count if self.chunk_count > 0 else 0,
            "chunks_per_second": self.chunk_count / streaming_time if streaming_time > 0 else 0
        } 