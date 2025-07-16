import pytest
import asyncio
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules to test
from langgraph_agents import LangGraphAgentWorkflow, AgentRole, AgentState
from agent_tools import AgentTools
from simple_agents import SimpleAgentWorkflow, AgentOpinion


class MockMultimodalLLM:
    """Mock multimodal LLM for testing"""
    
    def __init__(self):
        self.model = "test-model"
    
    def chat_with_chart(self, chart_path: str, prompt: str) -> Dict[str, Any]:
        return {
            "success": True,
            "analysis": f"Mock analysis for chart {chart_path} with prompt: {prompt[:100]}...",
            "model": self.model
        }
    
    def analyze_with_context(self, prompt: str) -> Dict[str, Any]:
        return {
            "success": True,
            "analysis": f"Mock context analysis: {prompt[:100]}...",
            "model": self.model
        }
    
    def analyze_chart_image(self, chart_path: str) -> Dict[str, Any]:
        return {
            "success": True,
            "analysis": f"Mock chart image analysis for {chart_path}",
            "model": self.model
        }


class MockAgentTools:
    """Mock agent tools for testing"""
    
    def __init__(self):
        self.available_tools = {
            "get_stock_data": self.get_stock_data,
            "get_financial_news": self.get_financial_news,
            "calculate_technical_indicators": self.calculate_technical_indicators,
            "get_market_sentiment": self.get_market_sentiment,
            "get_company_info": self.get_company_info,
        }
    
    def get_stock_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "current_price": 150.0,
            "previous_close": 148.0,
            "volume": 1000000,
            "price_change": 2.0,
            "price_change_pct": 1.35,
            "recent_data": {
                "dates": ["2025-01-15", "2025-01-16"],
                "close": [148.0, 150.0],
                "volume": [950000, 1000000]
            }
        }
    
    def get_financial_news(self, symbol: str, **kwargs) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "news_count": 5,
            "news": [
                {
                    "title": f"Mock news about {symbol}",
                    "summary": f"This is mock news content for {symbol}",
                    "publisher": "Mock Publisher",
                    "published": "2025-01-16"
                }
            ]
        }
    
    def calculate_technical_indicators(self, symbol: str, **kwargs) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "indicators": {
                "rsi": {"current": 45.0, "signal": "neutral"},
                "macd": {"macd": 0.5, "signal": 0.3, "signal_type": "bullish"},
                "sma": {"sma_20": 148.5, "sma_50": 145.0, "trend": "bullish"}
            }
        }
    
    def get_market_sentiment(self, symbol: str, **kwargs) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "news_sentiment": 0.2,
            "combined_sentiment": 0.15,
            "sentiment_label": "bullish",
            "confidence": 0.7
        }
    
    def get_company_info(self, symbol: str, **kwargs) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "company_name": f"Mock Company {symbol}",
            "sector": "Technology",
            "market_cap": 2000000000000,
            "pe_ratio": 25.5,
            "dividend_yield": 0.015
        }
    
    def calculate_support_resistance(self, symbol: str, **kwargs) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "current_price": 150.0,
            "resistance_levels": [155.0, 160.0],
            "support_levels": [145.0, 140.0]
        }
    
    def analyze_volume_profile(self, symbol: str, **kwargs) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "avg_volume": 800000,
            "recent_volume": 1000000,
            "volume_trend": "increasing"
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](**kwargs)
        return {"error": f"Tool {tool_name} not found"}


class TestAgentTools(unittest.TestCase):
    """Test agent tools functionality"""
    
    def setUp(self):
        self.agent_tools = MockAgentTools()
    
    def test_get_stock_data(self):
        """Test stock data retrieval"""
        result = self.agent_tools.get_stock_data("AAPL")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertIsInstance(result["current_price"], float)
        self.assertIsInstance(result["volume"], int)
        self.assertIn("recent_data", result)
    
    def test_get_financial_news(self):
        """Test financial news retrieval"""
        result = self.agent_tools.get_financial_news("AAPL")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertGreater(result["news_count"], 0)
        self.assertIn("news", result)
        self.assertIsInstance(result["news"], list)
    
    def test_calculate_technical_indicators(self):
        """Test technical indicators calculation"""
        result = self.agent_tools.calculate_technical_indicators("AAPL")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertIn("indicators", result)
        self.assertIn("rsi", result["indicators"])
        self.assertIn("macd", result["indicators"])
    
    def test_get_market_sentiment(self):
        """Test market sentiment analysis"""
        result = self.agent_tools.get_market_sentiment("AAPL")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertIn("sentiment_label", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)
    
    def test_execute_tool(self):
        """Test tool execution"""
        result = self.agent_tools.execute_tool("get_stock_data", symbol="AAPL")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertNotIn("error", result)
        
        # Test unknown tool
        result = self.agent_tools.execute_tool("unknown_tool")
        self.assertIn("error", result)


class TestSimpleAgentWorkflow(unittest.TestCase):
    """Test simple agent workflow"""
    
    def setUp(self):
        self.mock_llm = MockMultimodalLLM()
        self.mock_tools = MockAgentTools()
        self.workflow = SimpleAgentWorkflow(self.mock_llm, self.mock_tools)
    
    def test_extract_confidence(self):
        """Test confidence extraction from text"""
        text = "CONFIDENCE: 0.85"
        confidence = self.workflow._extract_confidence(text)
        self.assertEqual(confidence, 0.85)
        
        # Test default value
        text = "No confidence mentioned"
        confidence = self.workflow._extract_confidence(text)
        self.assertEqual(confidence, 0.5)
    
    def test_extract_recommendation(self):
        """Test recommendation extraction from text"""
        text = "RECOMMENDATION: BUY"
        recommendation = self.workflow._extract_recommendation(text)
        self.assertEqual(recommendation, "BUY")
        
        # Test default value
        text = "No recommendation mentioned"
        recommendation = self.workflow._extract_recommendation(text)
        self.assertEqual(recommendation, "HOLD")
    
    def test_extract_reasoning(self):
        """Test reasoning extraction from text"""
        text = "REASONING: This is a good buy because of strong fundamentals"
        reasoning = self.workflow._extract_reasoning(text)
        self.assertEqual(reasoning, "This is a good buy because of strong fundamentals")
    
    def test_calculate_agreement_level(self):
        """Test agreement level calculation"""
        opinions = [
            AgentOpinion("tech", AgentRole.TECHNICAL, "analysis", 0.8, "BUY", "reasoning", datetime.now()),
            AgentOpinion("fund", AgentRole.FUNDAMENTAL, "analysis", 0.7, "BUY", "reasoning", datetime.now()),
            AgentOpinion("news", AgentRole.NEWS, "analysis", 0.6, "HOLD", "reasoning", datetime.now())
        ]
        
        agreement = self.workflow._calculate_agreement_level(opinions)
        self.assertEqual(agreement, 2/3)  # 2 out of 3 agree on BUY
    
    @patch('asyncio.run')
    def test_collect_market_data(self, mock_asyncio):
        """Test market data collection"""
        # Run the actual async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(self.workflow._collect_market_data("AAPL"))
            
            self.assertIn("stock_data", result)
            self.assertIn("news", result)
            self.assertIn("technical_indicators", result)
            self.assertIn("sentiment", result)
            self.assertIn("company_info", result)
            
            # Verify data structure
            self.assertEqual(result["stock_data"]["symbol"], "AAPL")
            self.assertEqual(result["news"]["symbol"], "AAPL")
            
        finally:
            loop.close()
    
    def test_build_consensus(self):
        """Test consensus building"""
        opinions = [
            AgentOpinion("tech", AgentRole.TECHNICAL, "analysis", 0.8, "BUY", "reasoning", datetime.now()),
            AgentOpinion("fund", AgentRole.FUNDAMENTAL, "analysis", 0.7, "BUY", "reasoning", datetime.now()),
            AgentOpinion("news", AgentRole.NEWS, "analysis", 0.6, "SELL", "reasoning", datetime.now()),
            AgentOpinion("risk", AgentRole.RISK, "analysis", 0.5, "HOLD", "reasoning", datetime.now())
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            consensus = loop.run_until_complete(self.workflow._build_consensus(opinions))
            
            self.assertIn("recommendation", consensus)
            self.assertIn("confidence", consensus)
            self.assertIn("agreement_level", consensus)
            
            # BUY should win due to higher confidence scores
            self.assertEqual(consensus["recommendation"], "BUY")
            self.assertGreater(consensus["confidence"], 0)
            
        finally:
            loop.close()
    
    def test_generate_action_plan(self):
        """Test action plan generation"""
        # Test BUY consensus
        buy_consensus = {"recommendation": "BUY", "confidence": 0.8}
        action_plan = self.workflow._generate_action_plan(buy_consensus)
        
        self.assertEqual(action_plan["action"], "Consider long position")
        self.assertIn("entry_criteria", action_plan)
        self.assertIn("position_size", action_plan)
        
        # Test SELL consensus
        sell_consensus = {"recommendation": "SELL", "confidence": 0.7}
        action_plan = self.workflow._generate_action_plan(sell_consensus)
        
        self.assertEqual(action_plan["action"], "Consider short position or exit longs")
        
        # Test HOLD consensus
        hold_consensus = {"recommendation": "HOLD", "confidence": 0.5}
        action_plan = self.workflow._generate_action_plan(hold_consensus)
        
        self.assertEqual(action_plan["action"], "Stay on sidelines")


class TestLangGraphAgentWorkflow(unittest.TestCase):
    """Test LangGraph agent workflow"""
    
    def setUp(self):
        self.mock_llm = MockMultimodalLLM()
        self.mock_tools = MockAgentTools()
        
        # Try to create workflow, fall back to mock if LangGraph not available
        try:
            self.workflow = LangGraphAgentWorkflow(self.mock_llm, self.mock_tools)
        except ImportError:
            self.workflow = None
            self.skipTest("LangGraph not available")
    
    def test_agent_roles(self):
        """Test agent role enumeration"""
        roles = list(AgentRole)
        expected_roles = [AgentRole.TECHNICAL, AgentRole.FUNDAMENTAL, AgentRole.NEWS, AgentRole.RISK]
        
        for role in expected_roles:
            self.assertIn(role, roles)
    
    def test_parse_agent_response(self):
        """Test agent response parsing"""
        if self.workflow is None:
            self.skipTest("LangGraph not available")
        
        mock_response = {
            "analysis": "ANALYSIS: Strong bullish signals\nCONFIDENCE: 0.85\nRECOMMENDATION: BUY\nREASONING: Technical indicators show upward momentum"
        }
        
        parsed = self.workflow._parse_agent_response(mock_response, AgentRole.TECHNICAL)
        
        self.assertEqual(parsed["role"], AgentRole.TECHNICAL.value)
        self.assertEqual(parsed["confidence"], 0.85)
        self.assertEqual(parsed["recommendation"], "BUY")
        self.assertIn("Technical indicators", parsed["reasoning"])
    
    def test_check_consensus(self):
        """Test consensus checking"""
        if self.workflow is None:
            self.skipTest("LangGraph not available")
        
        # Test consensus reached
        recommendations = ["BUY", "BUY", "BUY", "HOLD"]
        consensus = self.workflow._check_consensus(recommendations)
        self.assertTrue(consensus)  # 75% agreement
        
        # Test no consensus
        recommendations = ["BUY", "SELL", "HOLD", "BUY"]
        consensus = self.workflow._check_consensus(recommendations)
        self.assertFalse(consensus)  # Only 50% agreement
    
    def test_build_weighted_consensus(self):
        """Test weighted consensus building"""
        if self.workflow is None:
            self.skipTest("LangGraph not available")
        
        opinions = [
            {"recommendation": "BUY", "confidence": 0.9},
            {"recommendation": "BUY", "confidence": 0.8},
            {"recommendation": "SELL", "confidence": 0.6},
            {"recommendation": "HOLD", "confidence": 0.5}
        ]
        
        consensus = self.workflow._build_weighted_consensus(opinions)
        
        self.assertEqual(consensus["recommendation"], "BUY")
        self.assertGreater(consensus["confidence"], 0.5)
        self.assertIn("weighted_votes", consensus)
    
    def test_is_us_security(self):
        """Test US security detection"""
        if self.workflow is None:
            self.skipTest("LangGraph not available")
        
        # Test US securities
        self.assertTrue(self.workflow._is_us_security("AAPL"))
        self.assertTrue(self.workflow._is_us_security("MSFT"))
        self.assertTrue(self.workflow._is_us_security("GOOGL"))
        
        # Test non-US securities (typically longer symbols)
        self.assertFalse(self.workflow._is_us_security("EURUSD"))
        self.assertFalse(self.workflow._is_us_security("GBPJPY"))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        self.mock_llm = MockMultimodalLLM()
        self.mock_tools = MockAgentTools()
        self.simple_workflow = SimpleAgentWorkflow(self.mock_llm, self.mock_tools)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run complete analysis
            result = loop.run_until_complete(
                self.simple_workflow.run_analysis("AAPL", "test_chart.png")
            )
            
            # Verify result structure
            self.assertTrue(result["success"])
            self.assertEqual(result["symbol"], "AAPL")
            self.assertIn("market_data", result)
            self.assertIn("agent_opinions", result)
            self.assertIn("consensus", result)
            self.assertIn("final_analysis", result)
            
            # Verify agent opinions
            opinions = result["agent_opinions"]
            self.assertIn("technical", opinions)
            self.assertIn("fundamental", opinions)
            self.assertIn("news", opinions)
            self.assertIn("risk", opinions)
            
            # Verify consensus
            consensus = result["consensus"]
            self.assertIn("recommendation", consensus)
            self.assertIn("confidence", consensus)
            self.assertIn(consensus["recommendation"], ["BUY", "SELL", "HOLD"])
            
            # Verify final analysis
            final_analysis = result["final_analysis"]
            self.assertIn("synthesis", final_analysis)
            self.assertIn("action_plan", final_analysis)
            self.assertIn("key_insights", final_analysis)
            
        finally:
            loop.close()
    
    def test_error_handling(self):
        """Test error handling in workflow"""
        # Create a workflow with failing tools
        class FailingTools:
            def get_stock_data(self, symbol):
                raise Exception("API Error")
            
            def get_financial_news(self, symbol):
                raise Exception("News API Error")
            
            def calculate_technical_indicators(self, symbol):
                raise Exception("Technical Analysis Error")
            
            def get_market_sentiment(self, symbol):
                raise Exception("Sentiment API Error")
            
            def get_company_info(self, symbol):
                raise Exception("Company Info Error")
            
            def calculate_support_resistance(self, symbol):
                raise Exception("Support/Resistance Error")
            
            def analyze_volume_profile(self, symbol):
                raise Exception("Volume Analysis Error")
        
        failing_workflow = SimpleAgentWorkflow(self.mock_llm, FailingTools())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                failing_workflow.run_analysis("AAPL", "test_chart.png")
            )
            
            # Should still return a result, but with errors
            self.assertFalse(result["success"])
            self.assertIn("error", result)
            
        finally:
            loop.close()
    
    def test_consensus_with_different_opinions(self):
        """Test consensus building with varying agent opinions"""
        # Create opinions with different recommendations
        opinions = [
            AgentOpinion("tech", AgentRole.TECHNICAL, "Strong buy signal", 0.9, "BUY", "RSI oversold", datetime.now()),
            AgentOpinion("fund", AgentRole.FUNDAMENTAL, "Overvalued", 0.8, "SELL", "High P/E ratio", datetime.now()),
            AgentOpinion("news", AgentRole.NEWS, "Positive sentiment", 0.7, "BUY", "Good earnings", datetime.now()),
            AgentOpinion("risk", AgentRole.RISK, "High volatility", 0.6, "HOLD", "Uncertain market", datetime.now())
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            consensus = loop.run_until_complete(self.simple_workflow._build_consensus(opinions))
            
            # BUY should win due to higher confidence scores (0.9 + 0.7 = 1.6 vs 0.8 for SELL)
            self.assertEqual(consensus["recommendation"], "BUY")
            self.assertLess(consensus["agreement_level"], 1.0)  # Not unanimous
            
        finally:
            loop.close()


def run_performance_test():
    """Performance test for the workflow"""
    print("Running performance test...")
    
    import time
    
    mock_llm = MockMultimodalLLM()
    mock_tools = MockAgentTools()
    workflow = SimpleAgentWorkflow(mock_llm, mock_tools)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    start_time = time.time()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        for symbol in symbols:
            result = loop.run_until_complete(workflow.run_analysis(symbol, "test_chart.png"))
            assert result["success"], f"Failed for {symbol}"
            
    finally:
        loop.close()
    
    end_time = time.time()
    
    print(f"Performance test completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per symbol: {(end_time - start_time) / len(symbols):.2f} seconds")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    run_performance_test()
    
    print("\nAll tests completed!")