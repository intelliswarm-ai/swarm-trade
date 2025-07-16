#!/usr/bin/env python3
"""
Test to verify the LLM RunTree validation error is fixed
"""

import os
from src.multimodal_llm import MultimodalLLM

def test_llm_fix():
    """Test that the LLM works without RunTree validation errors"""
    print("🧪 Testing LLM Fix for RunTree Validation Error")
    print("=" * 50)
    
    # Initialize the LLM
    llm = MultimodalLLM()
    
    # Test connection
    print("🔄 Testing Ollama connection...")
    connection_ok = llm.check_ollama_connection()
    
    if not connection_ok:
        print("❌ Ollama connection failed - server may not be running")
        print("💡 Start Ollama with: ollama serve")
        return False
    
    print("✅ Ollama connection successful")
    
    # Look for a recent screenshot to test with
    screenshot_dir = "screenshots"
    if os.path.exists(screenshot_dir):
        screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
        if screenshots:
            # Use the most recent screenshot
            latest_screenshot = sorted(screenshots)[-1]
            screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
            
            print(f"🖼️  Testing with screenshot: {screenshot_path}")
            print("🤖 Running LLM analysis (this may take a moment)...")
            
            try:
                # This should not throw RunTree validation errors
                result = llm.analyze_chart_image(screenshot_path)
                
                if result.get('success'):
                    print("✅ LLM analysis completed successfully!")
                    print("📊 Analysis preview:", result['analysis'][:100] + "...")
                    return True
                else:
                    print(f"⚠️  LLM analysis failed: {result.get('error', 'Unknown error')}")
                    return False
                    
            except Exception as e:
                print(f"❌ LLM analysis threw exception: {e}")
                return False
        else:
            print("⚠️  No screenshots found in screenshots directory")
            print("💡 Take a screenshot first with /screenshot command")
            return False
    else:
        print("⚠️  Screenshots directory not found")
        return False

if __name__ == "__main__":
    success = test_llm_fix()
    if success:
        print("\n🎉 LLM fix successful! RunTree validation error resolved.")
    else:
        print("\n⚠️  Test could not complete - check Ollama server or screenshots.")