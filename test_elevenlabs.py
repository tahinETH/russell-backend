#!/usr/bin/env python3
"""
ElevenLabs Text-to-Speech Test Script using ElevenLabsService
This script helps test the ElevenLabs service implementation
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

# Add the app directory to the path so we can import the service
sys.path.append(str(Path(__file__).parent / "app"))

from services.elevenlabs_service import ElevenLabsService
from config import settings

async def test_elevenlabs_service():
    """Test the ElevenLabsService implementation"""
    print("🔍 Testing ElevenLabs Service...")
    
    # Initialize the service
    service = ElevenLabsService()
    
    if not service.client:
        print("❌ ElevenLabs service not configured (missing API key)")
        return False
    
    print("✅ ElevenLabs service initialized successfully")
    print(f"📋 Using voice ID: {service.voice_id}")
    print(f"📋 Using model ID: {service.model_id}")
    
    return True, service

async def test_text_to_speech(service: ElevenLabsService):
    """Test text-to-speech conversion using the service"""
    print(f"\n🎤 Testing Text-to-Speech conversion...")
    
    test_text = "Hello! This is a test of ElevenLabs text-to-speech functionality using the service."
    
    try:
        print("🔄 Converting text to speech...")
        audio_data = await service.text_to_speech(
            text=test_text,
            output_format="mp3_44100_128"
        )
        
        if audio_data:
            # Save audio file
            with open('test_output.mp3', 'wb') as f:
                f.write(audio_data)
            print("✅ TTS successful! Audio saved as 'test_output.mp3'")
            
            # Show base64 encoded version (first 100 chars)
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            print(f"📝 Base64 (first 100 chars): {audio_base64[:100]}...")
            print(f"📊 Audio size: {len(audio_data)} bytes")
            return True
        else:
            print("❌ TTS failed: No audio data returned")
            return False
            
    except Exception as e:
        print(f"❌ TTS failed with error: {e}")
        return False

async def test_streaming_text_to_speech(service: ElevenLabsService):
    """Test streaming text-to-speech conversion"""
    print(f"\n🎤 Testing Streaming Text-to-Speech...")
    
    test_text = "This is a test of streaming text-to-speech functionality. The audio should be generated in chunks."
    
    try:
        print("🔄 Starting streaming TTS...")
        audio_chunks = []
        chunk_count = 0
        
        async for chunk in service.text_to_speech_stream(
            text=test_text,
            output_format="mp3_44100_128",
            chunk_size=1024
        ):
            if chunk:
                audio_chunks.append(chunk)
                chunk_count += 1
                print(f"📦 Received chunk {chunk_count} ({len(chunk)} bytes)")
        
        if audio_chunks:
            # Combine all chunks
            full_audio = b''.join(audio_chunks)
            
            # Save audio file
            with open('test_streaming_output.mp3', 'wb') as f:
                f.write(full_audio)
            
            print(f"✅ Streaming TTS successful! {chunk_count} chunks received")
            print(f"📊 Total audio size: {len(full_audio)} bytes")
            print("🎵 Audio saved as 'test_streaming_output.mp3'")
            return True
        else:
            print("❌ Streaming TTS failed: No audio chunks received")
            return False
            
    except Exception as e:
        print(f"❌ Streaming TTS failed with error: {e}")
        return False

async def test_different_voices(service: ElevenLabsService):
    """Test with different voice settings"""
    print(f"\n🎭 Testing different voice configurations...")
    
    # Test with different voice IDs if available
    test_voices = [
        "21m00Tcm4TlvDq8ikWAM",  # Rachel (default)
        "AZnzlk1XvdvUeBnXmlld",  # Domi
        "EXAVITQu4vr4xnSDxMaL",  # Bella
    ]
    
    for i, voice_id in enumerate(test_voices):
        try:
            print(f"\n🎤 Testing voice {i+1}: {voice_id}")
            audio_data = await service.text_to_speech(
                text=f"Hello, this is voice test number {i+1}.",
                voice_id=voice_id,
                output_format="mp3_44100_128"
            )
            
            if audio_data:
                filename = f'test_voice_{i+1}.mp3'
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                print(f"✅ Voice {i+1} successful! Saved as '{filename}'")
                print(f"📊 Audio size: {len(audio_data)} bytes")
            else:
                print(f"❌ Voice {i+1} failed: No audio data")
                
        except Exception as e:
            print(f"❌ Voice {i+1} failed: {e}")
            # Continue with next voice
            continue

def show_configuration():
    """Show current configuration"""
    print("\n⚙️  Current Configuration:")
    print(f"📋 API Key: {'✅ Set' if settings.elevenlabs_api_key else '❌ Missing'}")
    print(f"📋 Voice ID: {settings.elevenlabs_voice_id or 'Not set'}")
    print(f"📋 Model ID: {settings.elevenlabs_model_id or 'Not set'}")

async def main():
    print("🎯 ElevenLabs Service Test Script")
    print("=" * 50)
    
    # Show configuration
    show_configuration()
    
    # Test 1: Service initialization
    result = await test_elevenlabs_service()
    
    if isinstance(result, tuple) and len(result) == 2:
        success, service = result
        
        if success:
            # Test 2: Basic text-to-speech
            await test_text_to_speech(service)
            
            # Test 3: Streaming text-to-speech
            await test_streaming_text_to_speech(service)
            
            # Test 4: Different voices
            await test_different_voices(service)
        else:
            print("❌ Service initialization failed")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")
    print("\nGenerated files:")
    print("- test_output.mp3 (basic TTS)")
    print("- test_streaming_output.mp3 (streaming TTS)")
    print("- test_voice_*.mp3 (different voices)")

if __name__ == "__main__":
    asyncio.run(main()) 