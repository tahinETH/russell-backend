import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CustomerServiceLogger:
    def __init__(self, log_file_path: str = "customer_service_queries.jsonl"):
        """Initialize customer service query logger"""
        self.log_file_path = log_file_path
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else ".", exist_ok=True)
        
    def log_query(
        self,
        query: str,
        response: str,
        client_ip: Optional[str] = None,
        context_info: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Log a customer service query and response"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "client_ip": client_ip,
                "context_info": context_info,
                "response_time_ms": response_time_ms,
                "error": error,
                "service": "karseltex"
            }
            
            # Append to log file
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log customer service query: {e}")
    
    def get_recent_queries(self, limit: int = 100) -> list:
        """Get recent queries from the log file"""
        try:
            if not os.path.exists(self.log_file_path):
                return []
                
            queries = []
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # Get last 'limit' lines
            for line in lines[-limit:]:
                try:
                    queries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
                    
            return queries
            
        except Exception as e:
            logger.error(f"Failed to read customer service queries: {e}")
            return []
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get basic statistics about queries"""
        try:
            queries = self.get_recent_queries(limit=1000)  # Last 1000 queries
            
            if not queries:
                return {"total_queries": 0}
            
            total = len(queries)
            with_errors = len([q for q in queries if q.get("error")])
            avg_response_time = None
            
            response_times = [q.get("response_time_ms") for q in queries if q.get("response_time_ms")]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
            
            return {
                "total_queries": total,
                "queries_with_errors": with_errors,
                "error_rate": with_errors / total if total > 0 else 0,
                "avg_response_time_ms": avg_response_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get query statistics: {e}")
            return {"error": str(e)}