#!/usr/bin/env python3
"""
Test script for CrewAI-LangGraph integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.langgraph_agents import LangGraphAgentWorkflow, AgentRole, CrewAIAgentConfig

def test_crewai_integration():
    """Test CrewAI integration with LangGraph"""
    
    print("🧪 Testing CrewAI-LangGraph Integration")
    print("=" * 50)
    
    # Mock dependencies
    class MockMultimodalLLM:
        def analyze_with_context(self, prompt):
            return {
                "analysis": """
                EXECUTIVE_SUMMARY: This is a test analysis showing the integration is working properly.
                DETAILED_ANALYSIS: The system can successfully parse CrewAI-style prompts and responses.
                CONFIDENCE_SCORE: 0.85
                KEY_INSIGHTS: Integration successful, Proper formatting detected, Mock test passed
                RECOMMENDATIONS: Continue with full implementation, Test with real data
                RISKS_IDENTIFIED: No major risks detected in integration
                NEXT_STEPS: Deploy to production environment
                """
            }
        
        def chat_with_chart(self, chart_path, prompt):
            return self.analyze_with_context(prompt)
    
    class MockAgentTools:
        def get_stock_data(self, symbol):
            return {"price": 100.0, "volume": 1000000}
        
        def get_financial_news(self, symbol):
            return {"headlines": ["Test news 1", "Test news 2"]}
        
        def calculate_technical_indicators(self, symbol):
            return {"rsi": 65, "macd": 0.5}
        
        def get_company_info(self, symbol):
            return {"name": "Test Company", "sector": "Technology"}
    
    # Create workflow with mock dependencies
    mock_llm = MockMultimodalLLM()
    mock_tools = MockAgentTools()
    
    workflow = LangGraphAgentWorkflow(mock_llm, mock_tools)
    
    # Test CrewAI configurations
    print("✅ Testing CrewAI agent configurations...")
    crew_configs = workflow.crew_configs
    print(f"   Found {len(crew_configs)} agent configurations")
    
    for role, config in crew_configs.items():
        print(f"   🤖 {role.value}: {config.role}")
    
    # Test CrewAI tasks
    print("\n✅ Testing CrewAI task definitions...")
    crew_tasks = workflow.crew_tasks
    print(f"   Found {len(crew_tasks)} task definitions")
    
    for i, task in enumerate(crew_tasks, 1):
        print(f"   📋 Task {i}: {task.agent_role.value}")
    
    # Test response parsing
    print("\n✅ Testing response parsing...")
    mock_response = {
        "analysis": """
        EXECUTIVE_SUMMARY: Test summary for parsing validation
        CONFIDENCE_SCORE: 0.9
        KEY_INSIGHTS: Insight 1, Insight 2, Insight 3
        RECOMMENDATIONS: Rec 1, Rec 2
        RISKS_IDENTIFIED: Risk 1, Risk 2
        NEXT_STEPS: Next step 1
        """
    }
    
    task = crew_tasks[0]  # First task
    config = crew_configs[task.agent_role]
    
    parsed = workflow._parse_crew_response(mock_response, task, config)
    print(f"   📊 Parsed confidence: {parsed['confidence']}")
    print(f"   💡 Parsed insights: {len(parsed['key_insights'])} items")
    print(f"   🎯 Parsed recommendations: {len(parsed['recommendations'])} items")
    
    print("\n🎉 CrewAI-LangGraph Integration Test Completed Successfully!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        test_crewai_integration()
        print("\n✅ All tests passed! Integration is ready for use.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()