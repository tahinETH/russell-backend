import fal_client
import logging
from typing import Optional, AsyncGenerator, Dict, Any
import os
from ..config import settings

logger = logging.getLogger(__name__)

class FalService:
    """Service for generating images using Fal.ai"""
    
    def __init__(self):
        """Initialize the Fal service with API key from settings"""
        self.api_key = os.getenv("FAL_KEY")
        if self.api_key:
            # Configure fal_client with API key
            os.environ["FAL_KEY"] = self.api_key
            logger.info("Fal service initialized with API key")
        else:
            logger.warning("FAL_KEY not found in environment variables")
    
    async def generate_image_stream(
        self, 
        prompt: str, 
        model: str = "fal-ai/flux-general",
        image_size: str = "landscape_16_9",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images: int = 1,
        enable_safety_checker: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream image generation progress and return the final result
        
        Args:
            prompt: The image generation prompt
            model: The Fal model to use (default: flux-general)
            image_size: Image dimensions (default: landscape_16_9)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            num_images: Number of images to generate
            enable_safety_checker: Whether to enable safety checker
            
        Yields:
            Status updates and the final image result
        """
        if not self.api_key:
            logger.error("Cannot generate image: FAL_KEY not configured")
            yield {
                "type": "error",
                "error": "Image generation service not configured"
            }
            return
        
        try:
            logger.info(f"Starting image generation with prompt: {prompt[:100]}...")
            
            # Submit the image generation request
            handler = await fal_client.submit_async(
                model,
                arguments={
                    "prompt": prompt,
                    "image_size": image_size,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "num_images": num_images,
                    "enable_safety_checker": enable_safety_checker
                }
            )
            
            # Stream events as they come
            async for event in handler.iter_events(with_logs=True):
                if isinstance(event, dict):
                    # Process different event types
                    if event.get("type") == "log":
                        logger.debug(f"Fal log: {event.get('message', '')}")
                        yield {
                            "type": "progress",
                            "message": event.get("message", "Processing...")
                        }
                    elif event.get("type") == "status":
                        yield {
                            "type": "status",
                            "status": event.get("status", "unknown")
                        }
                    else:
                        # Forward any other event types
                        yield event
            
            # Get the final result
            result = await handler.get()
            
            # Extract image URLs from result
            images = result.get("images", [])
            if images:
                logger.info(f"Image generation completed. Generated {len(images)} image(s)")
                yield {
                    "type": "complete",
                    "images": images,
                    "prompt": prompt
                }
            else:
                logger.error("No images returned from Fal")
                yield {
                    "type": "error",
                    "error": "No images generated"
                }
                
        except Exception as e:
            logger.error(f"Error generating image with Fal: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
    async def generate_image(
        self, 
        prompt: str,
        model: str = "fal-ai/flux-general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an image and return the final result (non-streaming)
        
        Args:
            prompt: The image generation prompt
            model: The Fal model to use
            **kwargs: Additional arguments for the model
            
        Returns:
            Dict with image URLs or None if failed
        """
        if not self.api_key:
            logger.error("Cannot generate image: FAL_KEY not configured")
            return None
        
        try:
            logger.info(f"Generating image with prompt: {prompt[:100]}...")
            
            # Submit and wait for result
            result = await fal_client.run_async(
                model,
                arguments={
                    "prompt": prompt,
                    **kwargs
                }
            )
            
            images = result.get("images", [])
            if images:
                logger.info(f"Image generation completed. Generated {len(images)} image(s)")
                return {
                    "images": images,
                    "prompt": prompt
                }
            else:
                logger.error("No images returned from Fal")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image with Fal: {e}")
            return None
    
    async def check_status(self, model: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Check the status of an image generation request
        
        Args:
            model: The Fal model being used
            request_id: The request ID to check
            
        Returns:
            Status information or None if failed
        """
        try:
            status = await fal_client.status_async(model, request_id, with_logs=True)
            return status
        except Exception as e:
            logger.error(f"Error checking Fal status: {e}")
            return None 