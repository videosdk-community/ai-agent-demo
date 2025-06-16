#!/usr/bin/env python3
"""
Test script for different AI travel agent voices.
This script helps you test different voice configurations for your travel agent.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_MEETING_ID = "voice-test-meeting"
VIDEOSDK_TOKEN = "your-videosdk-token-here"  # Replace with your actual token

# Available voices to test
VOICES = [
    "Alloy",    # Neutral, professional
    "Echo",     # Warm, friendly
    "Fable",    # Clear, expressive
    "Onyx",     # Deep, authoritative
    "Nova",     # Bright, energetic (RECOMMENDED for travel)
    "Shimmer"   # Soft, gentle
]

# Travel agent system prompt
TRAVEL_AGENT_PROMPT = """You are a knowledgeable and friendly travel advisor AI assistant with access to a comprehensive database of 111 travel destinations worldwide.

## Your Role
Help users find perfect travel destinations by understanding their interests, budget, preferred time of year, and travel style. Use the vector store to search for relevant destinations and provide personalized recommendations.

## Key Capabilities
- Destination recommendations based on interests (beach, culture, history, nightlife, etc.)
- Travel planning advice including best times to visit
- Cultural insights and attraction information
- Practical travel tips and logistics

## Response Guidelines
1. Ask clarifying questions if needed (budget, interests, time of year, travel style)
2. Present 3-5 relevant destinations with brief descriptions
3. Include key attractions and best time to visit for each destination
4. Explain why each destination matches their interests
5. Be conversational and friendly - travel planning should be exciting!

Remember: You're helping people plan exciting adventures! Make the conversation engaging and helpful."""

def create_meeting_config(voice: str) -> Dict[str, Any]:
    """Create a meeting configuration with the specified voice."""
    return {
        "meeting_id": f"{TEST_MEETING_ID}-{voice.lower()}",
        "token": VIDEOSDK_TOKEN,
        "model": "gpt-4",
        "voice": voice,
        "personality": "travel_advisor",
        "temperature": 0.8,
        "system_prompt": TRAVEL_AGENT_PROMPT,
        "topP": 0.8,
        "topK": 40
    }

def test_voice(voice: str) -> bool:
    """Test a specific voice by joining an agent to a meeting."""
    print(f"\n🧪 Testing voice: {voice}")
    print("=" * 50)
    
    config = create_meeting_config(voice)
    
    try:
        # Join agent to meeting
        response = requests.post(
            f"{API_BASE_URL}/join-agent",
            json=config,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully joined agent with {voice} voice")
            print(f"   Meeting ID: {result.get('meeting_id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
            
            # Leave the agent after a short delay
            time.sleep(2)
            leave_response = requests.post(
                f"{API_BASE_URL}/leave-agent",
                json={"meeting_id": config["meeting_id"]},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if leave_response.status_code == 200:
                print(f"✅ Successfully left meeting")
            else:
                print(f"⚠️  Warning: Could not leave meeting properly")
            
            return True
            
        else:
            print(f"❌ Failed to join agent with {voice} voice")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error testing {voice} voice: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error testing {voice} voice: {e}")
        return False

def main():
    """Main function to test all voices."""
    print("🎤 AI Travel Agent Voice Testing")
    print("=" * 60)
    print("This script will test all available voices for your travel agent.")
    print("Make sure your server is running on localhost:8000")
    print("Update VIDEOSDK_TOKEN in the script with your actual token.")
    print()
    
    # Check if server is running
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server is not responding properly. Please check if it's running.")
            return
        print("✅ Server is running and responding")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Please make sure it's running on localhost:8000")
        return
    
    print(f"\n📋 Available voices to test: {', '.join(VOICES)}")
    print(f"⭐ Recommended for travel: Nova (bright and energetic)")
    
    # Test each voice
    successful_tests = 0
    for voice in VOICES:
        if test_voice(voice):
            successful_tests += 1
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total voices tested: {len(VOICES)}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {len(VOICES) - successful_tests}")
    
    if successful_tests == len(VOICES):
        print("🎉 All voices are working correctly!")
    elif successful_tests > 0:
        print("⚠️  Some voices are working, but there were issues with others.")
    else:
        print("❌ No voices are working. Please check your configuration.")
    
    print("\n💡 Next steps:")
    print("1. Choose your preferred voice from the successful tests")
    print("2. Update your frontend to use the selected voice")
    print("3. Test the voice with actual travel queries")

if __name__ == "__main__":
    main() 