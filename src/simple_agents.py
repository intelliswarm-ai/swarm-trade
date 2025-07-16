import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging

class AgentRole(Enum):
    TECHNICAL = "technical_analyst"
    FUNDAMENTAL = "fundamental_analyst"
    NEWS = "news_analyst"
    RISK = "risk_manager"
    MODERATOR = "moderator"

@dataclass
class AgentOpinion:
    agent_id: str
    role: AgentRole
    analysis: str
    confidence: float
    recommendation: str
    reasoning: str
    timestamp: datetime

class SimpleAgentWorkflow:
    """Simple agent workflow without LangGraph dependency"""
    
    def __init__(self, multimodal_llm, agent_tools):
        self.multimodal_llm = multimodal_llm
        self.agent_tools = agent_tools
        self.logger = logging.getLogger(__name__)
        
    async def run_analysis(self, symbol: str, chart_path: str) -> Dict[str, Any]:
        """Run the complete agentic analysis"""
        try:
            self.logger.info(f"Starting simple agentic analysis for {symbol}")
            
            # Phase 1: Data Collection
            print("📊 Phase 1: Collecting market data...")
            market_data = await self._collect_market_data(symbol)
            
            # Phase 2: Individual Agent Analysis
            print("🤖 Phase 2: Individual agent analysis...")
            agent_opinions = await self._get_agent_opinions(symbol, chart_path, market_data)
            
            # Phase 3: Debate and Consensus
            print("💬 Phase 3: Agent debate and consensus...")
            consensus = await self._build_consensus(agent_opinions)
            
            # Phase 4: Final Synthesis
            print("📝 Phase 4: Final synthesis...")
            final_analysis = await self._generate_final_analysis(consensus, symbol, agent_opinions)
            
            return {
                "success": True,
                "symbol": symbol,
                "market_data": market_data,
                "agent_opinions": {
                    "technical": next((op for op in agent_opinions if op.role == AgentRole.TECHNICAL), None),
                    "fundamental": next((op for op in agent_opinions if op.role == AgentRole.FUNDAMENTAL), None),
                    "news": next((op for op in agent_opinions if op.role == AgentRole.NEWS), None),
                    "risk": next((op for op in agent_opinions if op.role == AgentRole.RISK), None)
                },
                "consensus": consensus,
                "final_analysis": final_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in simple agentic analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _collect_market_data(self, symbol: str) -> Dict[str, Any]:
        """Collect market data using available tools"""
        market_data = {}
        
        try:
            # Stock data
            market_data["stock_data"] = self.agent_tools.get_stock_data(symbol)
            
            # Financial news
            market_data["news"] = self.agent_tools.get_financial_news(symbol)
            
            # Technical indicators
            market_data["technical_indicators"] = self.agent_tools.calculate_technical_indicators(symbol)
            
            # Market sentiment
            market_data["sentiment"] = self.agent_tools.get_market_sentiment(symbol)
            
            # Company info
            market_data["company_info"] = self.agent_tools.get_company_info(symbol)
            
            # Support/resistance levels
            market_data["support_resistance"] = self.agent_tools.calculate_support_resistance(symbol)
            
            # Volume analysis
            market_data["volume_analysis"] = self.agent_tools.analyze_volume_profile(symbol)
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
            market_data["error"] = str(e)
        
        return market_data
    
    async def _get_agent_opinions(self, symbol: str, chart_path: str, market_data: Dict) -> List[AgentOpinion]:
        """Get opinions from each agent"""
        opinions = []
        
        # Technical Analyst
        technical_opinion = await self._get_technical_opinion(symbol, chart_path, market_data)
        opinions.append(technical_opinion)
        
        # Fundamental Analyst
        fundamental_opinion = await self._get_fundamental_opinion(symbol, market_data)
        opinions.append(fundamental_opinion)
        
        # News Analyst
        news_opinion = await self._get_news_opinion(symbol, market_data)
        opinions.append(news_opinion)
        
        # Risk Manager
        risk_opinion = await self._get_risk_opinion(symbol, market_data)
        opinions.append(risk_opinion)
        
        return opinions
    
    async def _get_technical_opinion(self, symbol: str, chart_path: str, market_data: Dict) -> AgentOpinion:
        """Get technical analyst opinion"""
        try:
            # Filter relevant data
            relevant_data = {
                "stock_data": market_data.get("stock_data", {}),
                "technical_indicators": market_data.get("technical_indicators", {}),
                "support_resistance": market_data.get("support_resistance", {}),
                "volume_analysis": market_data.get("volume_analysis", {})
            }
            
            prompt = f"""
            You are a skilled technical analyst with 15+ years of experience analyzing {symbol}.
            
            Based on the chart and technical data:
            {json.dumps(relevant_data, indent=2, default=str)}
            
            Provide your analysis in this format:
            ANALYSIS: [Your detailed technical analysis]
            CONFIDENCE: [0.0 to 1.0]
            RECOMMENDATION: [BUY/SELL/HOLD]
            REASONING: [Your reasoning for the recommendation]
            
            Focus on chart patterns, indicators, support/resistance levels, and volume analysis.
            """
            
            # Get LLM response
            if chart_path:
                response = self.multimodal_llm.chat_with_chart(chart_path, prompt)
            else:
                response = self.multimodal_llm.analyze_with_context(prompt)
            
            return self._parse_agent_response(response, AgentRole.TECHNICAL, "technical_analyst")
            
        except Exception as e:
            self.logger.error(f"Error getting technical opinion: {str(e)}")
            return AgentOpinion(
                agent_id="technical_analyst",
                role=AgentRole.TECHNICAL,
                analysis=f"Error in technical analysis: {str(e)}",
                confidence=0.0,
                recommendation="HOLD",
                reasoning="Analysis failed",
                timestamp=datetime.now()
            )
    
    async def _get_fundamental_opinion(self, symbol: str, market_data: Dict) -> AgentOpinion:
        """Get fundamental analyst opinion"""
        try:
            relevant_data = {
                "company_info": market_data.get("company_info", {}),
                "stock_data": market_data.get("stock_data", {}),
            }
            
            prompt = f"""
            You are a fundamental analyst analyzing {symbol}.
            
            Based on the fundamental data:
            {json.dumps(relevant_data, indent=2, default=str)}
            
            Provide your analysis in this format:
            ANALYSIS: [Your detailed fundamental analysis]
            CONFIDENCE: [0.0 to 1.0]
            RECOMMENDATION: [BUY/SELL/HOLD]
            REASONING: [Your reasoning for the recommendation]
            
            Focus on valuation metrics, financial health, and growth prospects.
            """
            
            response = self.multimodal_llm.analyze_with_context(prompt)
            return self._parse_agent_response(response, AgentRole.FUNDAMENTAL, "fundamental_analyst")
            
        except Exception as e:
            self.logger.error(f"Error getting fundamental opinion: {str(e)}")
            return AgentOpinion(
                agent_id="fundamental_analyst",
                role=AgentRole.FUNDAMENTAL,
                analysis=f"Error in fundamental analysis: {str(e)}",
                confidence=0.0,
                recommendation="HOLD",
                reasoning="Analysis failed",
                timestamp=datetime.now()
            )
    
    async def _get_news_opinion(self, symbol: str, market_data: Dict) -> AgentOpinion:
        """Get news analyst opinion"""
        try:
            relevant_data = {
                "news": market_data.get("news", {}),
                "sentiment": market_data.get("sentiment", {})
            }
            
            prompt = f"""
            You are a news sentiment analyst analyzing {symbol}.
            
            Based on the news and sentiment data:
            {json.dumps(relevant_data, indent=2, default=str)}
            
            Provide your analysis in this format:
            ANALYSIS: [Your detailed sentiment analysis]
            CONFIDENCE: [0.0 to 1.0]
            RECOMMENDATION: [BUY/SELL/HOLD]
            REASONING: [Your reasoning for the recommendation]
            
            Focus on news sentiment, market psychology, and recent events.
            """
            
            response = self.multimodal_llm.analyze_with_context(prompt)
            return self._parse_agent_response(response, AgentRole.NEWS, "news_analyst")
            
        except Exception as e:
            self.logger.error(f"Error getting news opinion: {str(e)}")
            return AgentOpinion(
                agent_id="news_analyst",
                role=AgentRole.NEWS,
                analysis=f"Error in news analysis: {str(e)}",
                confidence=0.0,
                recommendation="HOLD",
                reasoning="Analysis failed",
                timestamp=datetime.now()
            )
    
    async def _get_risk_opinion(self, symbol: str, market_data: Dict) -> AgentOpinion:
        """Get risk manager opinion"""
        try:
            relevant_data = {
                "stock_data": market_data.get("stock_data", {}),
                "volume_analysis": market_data.get("volume_analysis", {}),
                "technical_indicators": market_data.get("technical_indicators", {})
            }
            
            prompt = f"""
            You are a risk manager analyzing {symbol}.
            
            Based on the risk data:
            {json.dumps(relevant_data, indent=2, default=str)}
            
            Provide your analysis in this format:
            ANALYSIS: [Your detailed risk analysis]
            CONFIDENCE: [0.0 to 1.0]
            RECOMMENDATION: [BUY/SELL/HOLD]
            REASONING: [Your reasoning for the recommendation]
            
            Focus on volatility, risk/reward ratios, and position sizing.
            """
            
            response = self.multimodal_llm.analyze_with_context(prompt)
            return self._parse_agent_response(response, AgentRole.RISK, "risk_manager")
            
        except Exception as e:
            self.logger.error(f"Error getting risk opinion: {str(e)}")
            return AgentOpinion(
                agent_id="risk_manager",
                role=AgentRole.RISK,
                analysis=f"Error in risk analysis: {str(e)}",
                confidence=0.0,
                recommendation="HOLD",
                reasoning="Analysis failed",
                timestamp=datetime.now()
            )
    
    def _parse_agent_response(self, response: Dict, role: AgentRole, agent_id: str) -> AgentOpinion:
        """Parse agent response into structured format"""
        try:
            if isinstance(response, dict):
                analysis_text = response.get("analysis", str(response))
            else:
                analysis_text = str(response)
            
            return AgentOpinion(
                agent_id=agent_id,
                role=role,
                analysis=analysis_text,
                confidence=self._extract_confidence(analysis_text),
                recommendation=self._extract_recommendation(analysis_text),
                reasoning=self._extract_reasoning(analysis_text),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing agent response: {str(e)}")
            return AgentOpinion(
                agent_id=agent_id,
                role=role,
                analysis="Error parsing response",
                confidence=0.0,
                recommendation="HOLD",
                reasoning="Parsing failed",
                timestamp=datetime.now()
            )
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        import re
        match = re.search(r'CONFIDENCE:\s*([0-9.]+)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation from text"""
        import re
        match = re.search(r'RECOMMENDATION:\s*(BUY|SELL|HOLD)', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return "HOLD"
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from text"""
        import re
        match = re.search(r'REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No reasoning provided"
    
    async def _build_consensus(self, opinions: List[AgentOpinion]) -> Dict[str, Any]:
        """Build consensus from agent opinions"""
        if not opinions:
            return {"recommendation": "HOLD", "confidence": 0.0}
        
        # Weight by confidence
        weighted_votes = {}
        total_weight = 0
        
        for opinion in opinions:
            rec = opinion.recommendation
            conf = opinion.confidence
            
            if rec not in weighted_votes:
                weighted_votes[rec] = 0
            weighted_votes[rec] += conf
            total_weight += conf
        
        # Get recommendation with highest weighted score
        if weighted_votes:
            final_rec = max(weighted_votes, key=weighted_votes.get)
            consensus_confidence = weighted_votes[final_rec] / total_weight if total_weight > 0 else 0
        else:
            final_rec = "HOLD"
            consensus_confidence = 0.0
        
        return {
            "recommendation": final_rec,
            "confidence": consensus_confidence,
            "weighted_votes": weighted_votes,
            "total_weight": total_weight,
            "agreement_level": self._calculate_agreement_level(opinions)
        }
    
    def _calculate_agreement_level(self, opinions: List[AgentOpinion]) -> float:
        """Calculate how much agents agree"""
        if not opinions:
            return 0.0
        
        from collections import Counter
        recommendations = [op.recommendation for op in opinions]
        counts = Counter(recommendations)
        most_common_count = counts.most_common(1)[0][1]
        
        return most_common_count / len(opinions)
    
    async def _generate_final_analysis(self, consensus: Dict, symbol: str, opinions: List[AgentOpinion]) -> Dict[str, Any]:
        """Generate final analysis"""
        try:
            # Create summary of all opinions
            opinions_summary = []
            for opinion in opinions:
                opinions_summary.append({
                    "agent": opinion.role.value,
                    "recommendation": opinion.recommendation,
                    "confidence": opinion.confidence,
                    "reasoning": opinion.reasoning[:200] + "..." if len(opinion.reasoning) > 200 else opinion.reasoning
                })
            
            # Create synthesis prompt
            synthesis_prompt = f"""
            As an experienced portfolio manager, synthesize the analysis for {symbol}.
            
            Agent Consensus: {consensus["recommendation"]} (Confidence: {consensus["confidence"]:.2f})
            Agreement Level: {consensus["agreement_level"]:.2f}
            
            Individual Agent Opinions:
            {json.dumps(opinions_summary, indent=2)}
            
            Provide a comprehensive final analysis including:
            1. Executive summary
            2. Key supporting factors
            3. Main risks
            4. Specific action plan
            5. Entry/exit criteria
            6. Position sizing recommendations
            
            Focus on practical, actionable insights for traders.
            """
            
            # Get final synthesis
            response = self.multimodal_llm.analyze_with_context(synthesis_prompt)
            
            if isinstance(response, dict):
                synthesis_text = response.get("analysis", str(response))
            else:
                synthesis_text = str(response)
            
            return {
                "recommendation": consensus["recommendation"],
                "confidence": consensus["confidence"],
                "agreement_level": consensus["agreement_level"],
                "synthesis": synthesis_text,
                "key_insights": self._extract_key_insights(opinions),
                "risk_assessment": self._extract_risk_assessment(opinions),
                "action_plan": self._generate_action_plan(consensus)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating final analysis: {str(e)}")
            return {
                "recommendation": consensus.get("recommendation", "HOLD"),
                "confidence": consensus.get("confidence", 0.0),
                "synthesis": f"Error generating synthesis: {str(e)}",
                "key_insights": [],
                "risk_assessment": {"level": "Unknown", "factors": []},
                "action_plan": {"action": "No action - analysis failed"}
            }
    
    def _extract_key_insights(self, opinions: List[AgentOpinion]) -> List[str]:
        """Extract key insights from opinions"""
        insights = []
        for opinion in opinions:
            # Extract key points from reasoning
            reasoning = opinion.reasoning
            if len(reasoning) > 50:  # Only include substantial reasoning
                insights.append(f"{opinion.role.value}: {reasoning[:150]}...")
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_risk_assessment(self, opinions: List[AgentOpinion]) -> Dict[str, Any]:
        """Extract risk assessment"""
        risk_opinion = next((op for op in opinions if op.role == AgentRole.RISK), None)
        
        if risk_opinion:
            return {
                "level": "High" if risk_opinion.confidence < 0.4 else "Medium" if risk_opinion.confidence < 0.7 else "Low",
                "factors": [risk_opinion.reasoning],
                "confidence": risk_opinion.confidence
            }
        
        return {
            "level": "Unknown",
            "factors": ["No risk analysis available"],
            "confidence": 0.0
        }
    
    def _generate_action_plan(self, consensus: Dict) -> Dict[str, Any]:
        """Generate action plan based on consensus"""
        recommendation = consensus["recommendation"]
        confidence = consensus["confidence"]
        
        if recommendation == "BUY":
            return {
                "action": "Consider long position",
                "entry_criteria": "Wait for confirmation signals",
                "position_size": f"{min(confidence * 2, 2):.1f}% of portfolio",
                "stop_loss": "Set below recent support",
                "take_profit": "Target next resistance level",
                "timeframe": "Monitor for trend continuation"
            }
        elif recommendation == "SELL":
            return {
                "action": "Consider short position or exit longs",
                "entry_criteria": "Wait for breakdown confirmation",
                "position_size": f"{min(confidence * 2, 2):.1f}% of portfolio",
                "stop_loss": "Set above recent resistance",
                "take_profit": "Target next support level",
                "timeframe": "Monitor for trend reversal"
            }
        else:
            return {
                "action": "Stay on sidelines",
                "entry_criteria": "Wait for clearer signals",
                "position_size": "0% - no position",
                "stop_loss": "N/A",
                "take_profit": "N/A",
                "timeframe": "Monitor for better setup"
            }