import os
import glob
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ContextService:
    """Service for loading and managing markdown file context for karseltex-int.com"""
    
    def __init__(self, markdown_directory: str = "data/karseltex_context"):
        """
        Initialize the context service
        
        Args:
            markdown_directory: Directory containing markdown files for context
        """
        self.markdown_directory = markdown_directory
        self.context_cache: Dict[str, str] = {}
        self.loaded_files: List[str] = []
    
    async def load_markdown_context(self) -> str:
        """
        Load all markdown files from the context directory and combine them into a single context string
        
        Returns:
            Combined markdown content as a string
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.markdown_directory, exist_ok=True)
            
            # Find all markdown files
            markdown_pattern = os.path.join(self.markdown_directory, "**/*.md")
            markdown_files = glob.glob(markdown_pattern, recursive=True)
            
            if not markdown_files:
                logger.warning(f"No markdown files found in {self.markdown_directory}")
                return "No context files available. Please add markdown files to the context directory."
            
            combined_context = []
            loaded_count = 0
            
            for file_path in sorted(markdown_files):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Add file separator and content
                            relative_path = os.path.relpath(file_path, self.markdown_directory)
                            combined_context.append(f"## File: {relative_path}\n\n{content}")
                            loaded_count += 1
                            
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    continue
            
            if not combined_context:
                return "No readable content found in markdown files."
            
            result = "\n\n---\n\n".join(combined_context)
            logger.info(f"Loaded {loaded_count} markdown files for context")
            
            # Cache the result
            self.context_cache["full_context"] = result
            self.loaded_files = markdown_files
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading markdown context: {e}")
            return f"Error loading context: {str(e)}"
    
    async def get_context(self, refresh: bool = False) -> str:
        """
        Get the markdown context, either from cache or by loading fresh
        
        Args:
            refresh: If True, reload files even if cached
            
        Returns:
            Combined markdown content as a string
        """
        if refresh or "full_context" not in self.context_cache:
            return await self.load_markdown_context()
        
        return self.context_cache.get("full_context", "")
    
    def get_loaded_files_info(self) -> Dict[str, any]:
        """
        Get information about loaded files
        
        Returns:
            Dictionary with file count and list of loaded files
        """
        return {
            "file_count": len(self.loaded_files),
            "files": [os.path.relpath(f, self.markdown_directory) for f in self.loaded_files],
            "context_size": len(self.context_cache.get("full_context", ""))
        }