from typing import Dict, List, Any, Optional, TypedDict
from tqdm import tqdm
import json
import asyncio
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

# Try to import LangGraph components with fallback
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Create mock classes for fallback
    class StateGraph:
        def __init__(self, state_class):
            self.state_class = state_class
            self.nodes = {}
            self.edges = {}
            self.entry_point = None
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)
        
        def add_conditional_edges(self, from_node, condition, mapping):
            pass
        
        def set_entry_point(self, node):
            self.entry_point = node
        
        def compile(self):
            return MockWorkflow(self)
    
    class MockWorkflow:
        def __init__(self, graph):
            self.graph = graph
        
        async def ainvoke(self, state):
            return await self._run_simplified_workflow(state)
        
        async def _run_simplified_workflow(self, state):
            # Simple sequential execution for fallback
            if 'data_collector' in self.graph.nodes:
                state = await self.graph.nodes['data_collector'](state)
            
            for node_name in ['technical_analyst', 'fundamental_analyst', 'news_analyst', 'risk_manager']:
                if node_name in self.graph.nodes:
                    state = await self.graph.nodes[node_name](state)
            
            if 'consensus_builder' in self.graph.nodes:
                state = await self.graph.nodes['consensus_builder'](state)
            
            if 'final_synthesizer' in self.graph.nodes:
                state = await self.graph.nodes['final_synthesizer'](state)
            
            return state
    
    class ToolNode:
        def __init__(self, tools):
            self.tools = tools
    
    class SystemMessage:
        def __init__(self, content):
            self.content = content
    
    class AIMessage:
        def __init__(self, content):
            self.content = content
    
    def tool(func):
        return func
    
    END = "END"

class AgentState(TypedDict):
    """State shared between all agents in the workflow"""
    symbol: str
    chart_path: str
    market_data: Dict[str, Any]
    technical_analysis: Optional[Dict[str, Any]]
    fundamental_analysis: Optional[Dict[str, Any]]
    news_analysis: Optional[Dict[str, Any]]
    risk_analysis: Optional[Dict[str, Any]]
    research_analysis: Optional[Dict[str, Any]]
    psychology_analysis: Optional[Dict[str, Any]]
    trading_strategy: Optional[Dict[str, Any]]
    debate_history: List[Dict[str, Any]]
    consensus: Optional[Dict[str, Any]]
    final_recommendation: Optional[Dict[str, Any]]
    current_round: int
    max_rounds: int
    messages: List[Any]
    crew_mode: bool
    agent_configs: Optional[Dict[str, Any]]
    task_results: Dict[str, Any]

class AgentRole(Enum):
    TECHNICAL = "technical_analyst"
    FUNDAMENTAL = "fundamental_analyst"
    NEWS = "news_analyst"
    RISK = "risk_manager"
    MODERATOR = "moderator"
    RESEARCH = "research_analyst"
    TRADING_STRATEGIST = "trading_strategist"
    MARKET_PSYCHOLOGIST = "market_psychologist"

@dataclass
class CrewAIAgentConfig:
    """CrewAI-style agent configuration"""
    role: str
    goal: str
    backstory: str
    verbose: bool = True
    allow_delegation: bool = False
    max_iter: int = 3
    memory: bool = True
    tools: List[str] = None

@dataclass
class CrewAITask:
    """CrewAI-style task definition"""
    description: str
    expected_output: str
    agent_role: AgentRole
    context: Dict[str, Any] = None
    tools: List[str] = None

class LangGraphAgentWorkflow:
    def __init__(self, multimodal_llm, data_tools):
        self.multimodal_llm = multimodal_llm
        self.data_tools = data_tools
        self.workflow = self._create_workflow()
        self.crew_configs = self._create_crew_configs()
        self.crew_tasks = self._create_crew_tasks()
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("data_collector", self._data_collector_node)
        workflow.add_node("technical_analyst", self._technical_analyst_node)
        workflow.add_node("fundamental_analyst", self._fundamental_analyst_node)
        workflow.add_node("news_analyst", self._news_analyst_node)
        workflow.add_node("risk_manager", self._risk_manager_node)
        workflow.add_node("debate_moderator", self._debate_moderator_node)
        workflow.add_node("consensus_builder", self._consensus_builder_node)
        workflow.add_node("final_synthesizer", self._final_synthesizer_node)
        
        # Add tools node
        tools = self._create_tools()
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)
        
        # Define the workflow edges
        workflow.set_entry_point("data_collector")
        
        workflow.add_edge("data_collector", "technical_analyst")
        workflow.add_edge("data_collector", "fundamental_analyst")
        workflow.add_edge("data_collector", "news_analyst")
        workflow.add_edge("data_collector", "risk_manager")
        
        workflow.add_edge("technical_analyst", "debate_moderator")
        workflow.add_edge("fundamental_analyst", "debate_moderator")
        workflow.add_edge("news_analyst", "debate_moderator")
        workflow.add_edge("risk_manager", "debate_moderator")
        
        workflow.add_conditional_edges(
            "debate_moderator",
            self._should_continue_debate,
            {
                "continue": "technical_analyst",
                "consensus": "consensus_builder"
            }
        )
        
        workflow.add_edge("consensus_builder", "final_synthesizer")
        workflow.add_edge("final_synthesizer", END)
        
        return workflow.compile()
    
    def _create_crew_configs(self) -> Dict[AgentRole, CrewAIAgentConfig]:
        """Create CrewAI-style agent configurations"""
        return {
            AgentRole.RESEARCH: CrewAIAgentConfig(
                role="Senior Research Analyst",
                goal="Gather comprehensive market intelligence and provide data-driven insights",
                backstory="""You are a seasoned research analyst with 15+ years of experience in financial markets.
                You excel at gathering, synthesizing, and interpreting complex market data from multiple sources.
                Your analytical approach is methodical and data-driven, and you're known for uncovering insights
                that others miss. You have access to advanced research tools and databases.""",
                tools=["market_data", "news_scraper", "research_database"]
            ),
            
            AgentRole.TECHNICAL: CrewAIAgentConfig(
                role="Expert Technical Analyst",
                goal="Analyze price action, chart patterns, and technical indicators to predict market movements",
                backstory="""You are a world-class technical analyst with deep expertise in chart analysis,
                pattern recognition, and technical indicators. You've developed proprietary methods for
                identifying market trends and turning points. Your analysis is precise, actionable, and
                backed by decades of market observation.""",
                tools=["chart_analysis", "technical_indicators", "pattern_recognition"]
            ),
            
            AgentRole.FUNDAMENTAL: CrewAIAgentConfig(
                role="Fundamental Analysis Expert",
                goal="Evaluate company financials, industry trends, and economic factors to determine intrinsic value",
                backstory="""You are a fundamental analyst with a PhD in Finance and 20+ years of experience
                analyzing companies across various sectors. You excel at financial modeling, valuation,
                and identifying value opportunities. Your approach is thorough, conservative, and focused
                on long-term value creation.""",
                tools=["financial_data", "valuation_models", "sector_analysis"]
            ),
            
            AgentRole.RISK: CrewAIAgentConfig(
                role="Chief Risk Officer",
                goal="Assess and quantify risks, ensure proper risk management, and protect capital",
                backstory="""You are a risk management expert with extensive experience in portfolio
                management and risk assessment. You've navigated multiple market crashes and understand
                how to protect capital while maximizing returns. Your approach is systematic, quantitative,
                and focused on downside protection.""",
                tools=["risk_metrics", "correlation_analysis", "stress_testing"]
            ),
            
            AgentRole.MARKET_PSYCHOLOGIST: CrewAIAgentConfig(
                role="Market Psychology Specialist",
                goal="Analyze market sentiment, behavior patterns, and psychological factors",
                backstory="""You are a market psychology expert with a background in behavioral finance.
                You understand how emotions, biases, and crowd behavior drive market movements.
                Your analysis focuses on sentiment indicators, market positioning, and psychological
                turning points.""",
                tools=["sentiment_analysis", "social_media_monitoring", "options_flow"]
            ),
            
            AgentRole.TRADING_STRATEGIST: CrewAIAgentConfig(
                role="Senior Trading Strategist",
                goal="Develop actionable trading strategies and execution plans",
                backstory="""You are a veteran trading strategist with 25+ years of experience across
                multiple asset classes. You excel at translating analysis into practical trading strategies
                with clear entry/exit criteria. Your strategies are battle-tested and designed for
                real-world market conditions.""",
                tools=["strategy_builder", "backtesting", "execution_algorithms"]
            )
        }
    
    def _create_crew_tasks(self) -> List[CrewAITask]:
        """Create CrewAI-style tasks"""
        return [
            CrewAITask(
                description="Gather comprehensive market data, news, and research for the given symbol",
                expected_output="Detailed market intelligence report with key data points and insights",
                agent_role=AgentRole.RESEARCH,
                tools=["market_data", "news_scraper", "research_database"]
            ),
            
            CrewAITask(
                description="Perform detailed technical analysis including chart patterns, indicators, and price action",
                expected_output="Technical analysis report with specific entry/exit levels and pattern identification",
                agent_role=AgentRole.TECHNICAL,
                tools=["chart_analysis", "technical_indicators", "pattern_recognition"]
            ),
            
            CrewAITask(
                description="Conduct fundamental analysis including financial metrics, valuation, and company health",
                expected_output="Fundamental analysis report with valuation assessment and financial health score",
                agent_role=AgentRole.FUNDAMENTAL,
                tools=["financial_data", "valuation_models", "sector_analysis"]
            ),
            
            CrewAITask(
                description="Analyze market sentiment, news sentiment, and psychological factors",
                expected_output="Market psychology report with sentiment analysis and behavioral insights",
                agent_role=AgentRole.MARKET_PSYCHOLOGIST,
                tools=["sentiment_analysis", "social_media_monitoring", "options_flow"]
            ),
            
            CrewAITask(
                description="Assess all risk factors including market risk, company-specific risk, and portfolio impact",
                expected_output="Comprehensive risk assessment with risk metrics and mitigation strategies",
                agent_role=AgentRole.RISK,
                tools=["risk_metrics", "correlation_analysis", "stress_testing"]
            ),
            
            CrewAITask(
                description="Synthesize all analysis into actionable trading strategies with clear execution plans",
                expected_output="Trading strategy document with entry/exit criteria, position sizing, and risk management",
                agent_role=AgentRole.TRADING_STRATEGIST,
                tools=["strategy_builder", "backtesting", "execution_algorithms"]
            )
        ]
    
    def _create_tools(self) -> List:
        """Create tools for the agents to use"""
        
        @tool
        async def fetch_news(symbol: str) -> Dict[str, Any]:
            """Fetch news for a given symbol"""
            try:
                if hasattr(self.data_tools, 'get_news_for_symbol'):
                    return await self.data_tools.get_news_for_symbol(symbol)
                else:
                    return self.data_tools.get_financial_news(symbol)
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        async def fetch_financial_data(symbol: str) -> Dict[str, Any]:
            """Fetch financial data for a given symbol"""
            try:
                if hasattr(self.data_tools, 'get_financial_data'):
                    return await self.data_tools.get_financial_data(symbol)
                else:
                    return self.data_tools.get_company_info(symbol)
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        async def fetch_sec_filings(symbol: str) -> Dict[str, Any]:
            """Fetch SEC filings for a given symbol"""
            try:
                if hasattr(self.data_tools, 'get_sec_filings'):
                    return await self.data_tools.get_sec_filings(symbol)
                else:
                    return {"error": "SEC filings not available"}
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        async def fetch_yahoo_finance_data(symbol: str) -> Dict[str, Any]:
            """Fetch Yahoo Finance data for a given symbol"""
            try:
                if hasattr(self.data_tools, 'get_yahoo_finance_data'):
                    return await self.data_tools.get_yahoo_finance_data(symbol)
                else:
                    return self.data_tools.get_stock_data(symbol)
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        async def analyze_chart_image(chart_path: str) -> Dict[str, Any]:
            """Analyze chart image for technical patterns"""
            try:
                if hasattr(self.data_tools, 'analyze_chart_image'):
                    return await self.data_tools.analyze_chart_image(chart_path)
                else:
                    return {"error": "Chart image analysis not available"}
            except Exception as e:
                return {"error": str(e)}
        
        return [fetch_news, fetch_financial_data, fetch_sec_filings, fetch_yahoo_finance_data, analyze_chart_image]
    
    async def _data_collector_node(self, state: AgentState) -> AgentState:
        """Collect all market data for the symbol"""
        symbol = state["symbol"]
        chart_path = state["chart_path"]
        
        # Collect data from all sources
        market_data = {}
        
        # Chart analysis
        try:
            if hasattr(self.data_tools, 'analyze_chart_image'):
                market_data["chart_analysis"] = await self.data_tools.analyze_chart_image(chart_path)
            else:
                market_data["chart_analysis"] = {"error": "Chart analysis not available"}
        except Exception as e:
            market_data["chart_analysis"] = {"error": str(e)}
        
        # News data
        try:
            if hasattr(self.data_tools, 'get_news_for_symbol'):
                market_data["news"] = await self.data_tools.get_news_for_symbol(symbol)
            else:
                market_data["news"] = self.data_tools.get_financial_news(symbol)
        except Exception as e:
            market_data["news"] = {"error": str(e)}
        
        # Financial data
        try:
            if hasattr(self.data_tools, 'get_financial_data'):
                market_data["financial_data"] = await self.data_tools.get_financial_data(symbol)
            else:
                market_data["financial_data"] = self.data_tools.get_company_info(symbol)
        except Exception as e:
            market_data["financial_data"] = {"error": str(e)}
        
        # SEC filings (if US security)
        if self._is_us_security(symbol):
            try:
                if hasattr(self.data_tools, 'get_sec_filings'):
                    market_data["sec_filings"] = await self.data_tools.get_sec_filings(symbol)
                else:
                    market_data["sec_filings"] = {"error": "SEC filings not available"}
            except Exception as e:
                market_data["sec_filings"] = {"error": str(e)}
        
        # Yahoo Finance data
        try:
            if hasattr(self.data_tools, 'get_yahoo_finance_data'):
                market_data["yahoo_data"] = await self.data_tools.get_yahoo_finance_data(symbol)
            else:
                market_data["yahoo_data"] = self.data_tools.get_stock_data(symbol)
        except Exception as e:
            market_data["yahoo_data"] = {"error": str(e)}
        
        state["market_data"] = market_data
        state["messages"].append(SystemMessage(content=f"Data collection completed for {symbol}"))
        
        return state
    
    async def _technical_analyst_node(self, state: AgentState) -> AgentState:
        """Technical analyst performs chart analysis"""
        symbol = state["symbol"]
        chart_path = state["chart_path"]
        market_data = state["market_data"]
        
        # Filter relevant data for technical analysis
        technical_data = {
            "chart_analysis": market_data.get("chart_analysis", {}),
            "price_data": market_data.get("yahoo_data", {}).get("price_data", {}),
            "volume_data": market_data.get("yahoo_data", {}).get("volume_data", {}),
        }
        
        # Create technical analyst prompt
        prompt = f"""
        You are a skilled technical analyst with 15+ years of experience. 
        You focus on chart patterns, indicators, support/resistance levels, and price action.
        You are analytical, data-driven, and prefer quantitative analysis over speculation.
        
        Analyze {symbol} based on the chart and technical data:
        {json.dumps(technical_data, indent=2)}
        
        Provide your analysis in this format:
        ANALYSIS: [Your detailed technical analysis]
        CONFIDENCE: [0.0 to 1.0]
        RECOMMENDATION: [BUY/SELL/HOLD]
        REASONING: [Your reasoning for the recommendation]
        KEY_PATTERNS: [List of key patterns identified]
        SUPPORT_LEVELS: [Key support levels]
        RESISTANCE_LEVELS: [Key resistance levels]
        ENTRY_POINTS: [Suggested entry points]
        STOP_LOSS: [Stop loss recommendations]
        TARGET_PRICE: [Price targets]
        """
        
        # Get analysis from multimodal LLM
        try:
            response = await self._async_llm_call(chart_path, prompt)
        except Exception as e:
            response = {"analysis": f"Error in technical analysis: {str(e)}", "success": False}
        
        # Parse and structure the response
        technical_analysis = self._parse_agent_response(response, AgentRole.TECHNICAL)
        
        state["technical_analysis"] = technical_analysis
        state["messages"].append(AIMessage(content=f"Technical Analysis: {technical_analysis['recommendation']} (Confidence: {technical_analysis['confidence']})"))
        
        return state
    
    async def _fundamental_analyst_node(self, state: AgentState) -> AgentState:
        """Fundamental analyst performs financial analysis"""
        symbol = state["symbol"]
        market_data = state["market_data"]
        
        # Filter relevant data for fundamental analysis
        fundamental_data = {
            "financial_data": market_data.get("financial_data", {}),
            "sec_filings": market_data.get("sec_filings", {}),
            "earnings_data": market_data.get("yahoo_data", {}).get("earnings", {}),
            "fundamentals": market_data.get("yahoo_data", {}).get("fundamentals", {}),
        }
        
        # Create fundamental analyst prompt
        prompt = f"""
        You are a fundamental analyst with deep knowledge of financial markets.
        You analyze company financials, economic indicators, earnings, and business fundamentals.
        You are thorough, methodical, and focus on long-term value over short-term price movements.
        
        Analyze {symbol} based on the fundamental data:
        {json.dumps(fundamental_data, indent=2)}
        
        Provide your analysis in this format:
        ANALYSIS: [Your detailed fundamental analysis]
        CONFIDENCE: [0.0 to 1.0]
        RECOMMENDATION: [BUY/SELL/HOLD]
        REASONING: [Your reasoning for the recommendation]
        VALUATION: [Fair value assessment]
        GROWTH_PROSPECTS: [Growth analysis]
        FINANCIAL_HEALTH: [Financial strength assessment]
        COMPETITIVE_POSITION: [Market position analysis]
        RISKS: [Key fundamental risks]
        CATALYSTS: [Potential catalysts]
        """
        
        # Get analysis from LLM
        response = await self.multimodal_llm.analyze_with_context(prompt)
        
        # Parse and structure the response
        fundamental_analysis = self._parse_agent_response(response, AgentRole.FUNDAMENTAL)
        
        state["fundamental_analysis"] = fundamental_analysis
        state["messages"].append(AIMessage(content=f"Fundamental Analysis: {fundamental_analysis['recommendation']} (Confidence: {fundamental_analysis['confidence']})"))
        
        return state
    
    async def _news_analyst_node(self, state: AgentState) -> AgentState:
        """News analyst performs sentiment analysis"""
        symbol = state["symbol"]
        market_data = state["market_data"]
        
        # Filter relevant data for news analysis
        news_data = {
            "news": market_data.get("news", {}),
            "sentiment_data": market_data.get("yahoo_data", {}).get("sentiment", {}),
        }
        
        # Create news analyst prompt
        prompt = f"""
        You are a news and sentiment analyst who specializes in market psychology.
        You analyze news sentiment, social media trends, and market sentiment indicators.
        You are quick to identify sentiment shifts and their potential market impact.
        
        Analyze {symbol} based on the news and sentiment data:
        {json.dumps(news_data, indent=2)}
        
        Provide your analysis in this format:
        ANALYSIS: [Your detailed sentiment analysis]
        CONFIDENCE: [0.0 to 1.0]
        RECOMMENDATION: [BUY/SELL/HOLD]
        REASONING: [Your reasoning for the recommendation]
        SENTIMENT_SCORE: [Overall sentiment score]
        NEWS_IMPACT: [Impact of recent news]
        MARKET_PSYCHOLOGY: [Market psychology assessment]
        CATALYSTS: [Upcoming events or catalysts]
        SENTIMENT_TRENDS: [Sentiment trend analysis]
        """
        
        # Get analysis from LLM
        response = await self.multimodal_llm.analyze_with_context(prompt)
        
        # Parse and structure the response
        news_analysis = self._parse_agent_response(response, AgentRole.NEWS)
        
        state["news_analysis"] = news_analysis
        state["messages"].append(AIMessage(content=f"News Analysis: {news_analysis['recommendation']} (Confidence: {news_analysis['confidence']})"))
        
        return state
    
    async def _risk_manager_node(self, state: AgentState) -> AgentState:
        """Risk manager performs risk assessment"""
        symbol = state["symbol"]
        market_data = state["market_data"]
        
        # Filter relevant data for risk analysis
        risk_data = {
            "volatility_data": market_data.get("yahoo_data", {}).get("volatility", {}),
            "risk_metrics": market_data.get("yahoo_data", {}).get("risk_metrics", {}),
            "correlation_data": market_data.get("yahoo_data", {}).get("correlations", {}),
        }
        
        # Create risk manager prompt
        prompt = f"""
        You are a risk management specialist focused on capital preservation.
        You analyze potential risks, calculate position sizes, and assess risk/reward ratios.
        You are conservative, pragmatic, and often play devil's advocate to other analysts.
        
        Analyze {symbol} based on the risk data:
        {json.dumps(risk_data, indent=2)}
        
        Provide your analysis in this format:
        ANALYSIS: [Your detailed risk analysis]
        CONFIDENCE: [0.0 to 1.0]
        RECOMMENDATION: [BUY/SELL/HOLD]
        REASONING: [Your reasoning for the recommendation]
        RISK_LEVEL: [LOW/MEDIUM/HIGH]
        VOLATILITY_ASSESSMENT: [Volatility analysis]
        POSITION_SIZE: [Recommended position size]
        RISK_REWARD_RATIO: [Risk/reward assessment]
        DRAWDOWN_RISK: [Potential drawdown analysis]
        CORRELATION_RISKS: [Correlation risk assessment]
        """
        
        # Get analysis from LLM
        response = await self.multimodal_llm.analyze_with_context(prompt)
        
        # Parse and structure the response
        risk_analysis = self._parse_agent_response(response, AgentRole.RISK)
        
        state["risk_analysis"] = risk_analysis
        state["messages"].append(AIMessage(content=f"Risk Analysis: {risk_analysis['recommendation']} (Confidence: {risk_analysis['confidence']})"))
        
        return state
    
    async def _debate_moderator_node(self, state: AgentState) -> AgentState:
        """Moderator facilitates debate between agents"""
        current_round = state.get("current_round", 0)
        max_rounds = state.get("max_rounds", 3)
        
        # Collect all agent opinions
        opinions = []
        if state.get("technical_analysis"):
            opinions.append(("Technical Analyst", state["technical_analysis"]))
        if state.get("fundamental_analysis"):
            opinions.append(("Fundamental Analyst", state["fundamental_analysis"]))
        if state.get("news_analysis"):
            opinions.append(("News Analyst", state["news_analysis"]))
        if state.get("risk_analysis"):
            opinions.append(("Risk Manager", state["risk_analysis"]))
        
        # Check for consensus
        recommendations = [opinion[1]["recommendation"] for opinion in opinions]
        consensus_reached = self._check_consensus(recommendations)
        
        if consensus_reached or current_round >= max_rounds:
            state["consensus"] = {
                "reached": True,
                "round": current_round,
                "opinions": opinions
            }
            return state
        
        # Facilitate debate
        debate_prompt = self._create_debate_prompt(opinions, current_round)
        
        # Record debate round
        debate_round = {
            "round": current_round + 1,
            "opinions": opinions,
            "consensus_reached": consensus_reached,
            "debate_topic": self._identify_disagreement(recommendations)
        }
        
        if "debate_history" not in state:
            state["debate_history"] = []
        state["debate_history"].append(debate_round)
        
        state["current_round"] = current_round + 1
        state["messages"].append(SystemMessage(content=f"Debate round {current_round + 1}: {debate_round['debate_topic']}"))
        
        return state
    
    def _should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue or if consensus is reached"""
        consensus = state.get("consensus", {})
        
        if consensus.get("reached", False):
            return "consensus"
        
        current_round = state.get("current_round", 0)
        max_rounds = state.get("max_rounds", 3)
        
        if current_round >= max_rounds:
            return "consensus"
        
        return "continue"
    
    async def _consensus_builder_node(self, state: AgentState) -> AgentState:
        """Build consensus from agent opinions"""
        opinions = []
        
        if state.get("technical_analysis"):
            opinions.append(state["technical_analysis"])
        if state.get("fundamental_analysis"):
            opinions.append(state["fundamental_analysis"])
        if state.get("news_analysis"):
            opinions.append(state["news_analysis"])
        if state.get("risk_analysis"):
            opinions.append(state["risk_analysis"])
        
        # Build weighted consensus
        consensus = self._build_weighted_consensus(opinions)
        
        state["consensus"] = consensus
        state["messages"].append(SystemMessage(content=f"Consensus reached: {consensus['recommendation']} (Confidence: {consensus['confidence']:.2f})"))
        
        return state
    
    async def _final_synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final recommendation"""
        symbol = state["symbol"]
        consensus = state["consensus"]
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        As an experienced portfolio manager, synthesize the analysis for {symbol}.
        
        Consensus: {consensus['recommendation']} (Confidence: {consensus['confidence']:.2f})
        
        Individual Agent Opinions:
        Technical: {state.get('technical_analysis', {}).get('recommendation', 'N/A')}
        Fundamental: {state.get('fundamental_analysis', {}).get('recommendation', 'N/A')}
        News: {state.get('news_analysis', {}).get('recommendation', 'N/A')}
        Risk: {state.get('risk_analysis', {}).get('recommendation', 'N/A')}
        
        Provide a comprehensive final analysis with:
        1. Executive summary
        2. Key supporting factors
        3. Main risks
        4. Specific action plan
        5. Entry/exit criteria
        6. Position sizing recommendations
        """
        
        # Get final synthesis
        response = await self.multimodal_llm.analyze_with_context(synthesis_prompt)
        
        final_recommendation = {
            "recommendation": consensus["recommendation"],
            "confidence": consensus["confidence"],
            "synthesis": response.get("analysis", ""),
            "timestamp": datetime.now().isoformat(),
            "agent_opinions": {
                "technical": state.get("technical_analysis"),
                "fundamental": state.get("fundamental_analysis"),
                "news": state.get("news_analysis"),
                "risk": state.get("risk_analysis")
            },
            "debate_history": state.get("debate_history", [])
        }
        
        state["final_recommendation"] = final_recommendation
        state["messages"].append(AIMessage(content=f"Final Recommendation: {final_recommendation['recommendation']}"))
        
        return state
    
    def _parse_agent_response(self, response: Dict, role: AgentRole) -> Dict:
        """Parse agent response into structured format"""
        analysis_text = response.get("analysis", "")
        
        return {
            "role": role.value,
            "analysis": analysis_text,
            "confidence": self._extract_confidence(analysis_text),
            "recommendation": self._extract_recommendation(analysis_text),
            "reasoning": self._extract_reasoning(analysis_text),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        import re
        match = re.search(r'CONFIDENCE:\s*([0-9.]+)', text)
        return float(match.group(1)) if match else 0.5
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation from text"""
        import re
        match = re.search(r'RECOMMENDATION:\s*(BUY|SELL|HOLD)', text)
        return match.group(1) if match else "HOLD"
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from text"""
        import re
        match = re.search(r'REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.DOTALL)
        return match.group(1).strip() if match else "No reasoning provided"
    
    def _check_consensus(self, recommendations: List[str]) -> bool:
        """Check if recommendations have reached consensus"""
        from collections import Counter
        counts = Counter(recommendations)
        most_common = counts.most_common(1)[0]
        return most_common[1] >= len(recommendations) * 0.6
    
    def _identify_disagreement(self, recommendations: List[str]) -> str:
        """Identify main disagreement topic"""
        unique_recs = set(recommendations)
        if len(unique_recs) > 1:
            return f"Disagreement on recommendation: {', '.join(unique_recs)}"
        return "Minor disagreements on confidence levels"
    
    def _create_debate_prompt(self, opinions: List, round_num: int) -> str:
        """Create debate prompt for agents"""
        opinions_text = "\n".join([
            f"{name}: {opinion['recommendation']} (Confidence: {opinion['confidence']:.2f})"
            for name, opinion in opinions
        ])
        
        return f"""
        Round {round_num} Debate Topic: Resolve disagreements
        
        Current Opinions:
        {opinions_text}
        
        Each agent should consider others' viewpoints and either:
        1. Defend their position with additional evidence
        2. Adjust their position based on compelling arguments
        3. Find middle ground where appropriate
        """
    
    def _build_weighted_consensus(self, opinions: List[Dict]) -> Dict:
        """Build weighted consensus from opinions"""
        if not opinions:
            return {"recommendation": "HOLD", "confidence": 0.0}
        
        # Weight by confidence
        weighted_votes = {}
        total_weight = 0
        
        for opinion in opinions:
            rec = opinion["recommendation"]
            conf = opinion["confidence"]
            
            if rec not in weighted_votes:
                weighted_votes[rec] = 0
            weighted_votes[rec] += conf
            total_weight += conf
        
        # Get recommendation with highest weighted score
        final_rec = max(weighted_votes, key=weighted_votes.get)
        
        # Calculate consensus confidence
        consensus_confidence = weighted_votes[final_rec] / total_weight
        
        return {
            "recommendation": final_rec,
            "confidence": consensus_confidence,
            "weighted_votes": weighted_votes,
            "total_weight": total_weight
        }
    
    def _is_us_security(self, symbol: str) -> bool:
        """Check if symbol is a US security"""
        return len(symbol) <= 5 and symbol.isalpha()
    
    async def _async_llm_call(self, chart_path: str, prompt: str) -> Dict[str, Any]:
        """Async wrapper for LLM calls"""
        try:
            if chart_path:
                # For chart-based analysis
                if hasattr(self.multimodal_llm, 'chat_with_chart'):
                    return self.multimodal_llm.chat_with_chart(chart_path, prompt)
                else:
                    return self.multimodal_llm.analyze_chart_image(chart_path, prompt)
            else:
                # For text-based analysis - use chart analysis with a dummy image or create text-only analysis
                if hasattr(self.multimodal_llm, 'analyze_with_context'):
                    return self.multimodal_llm.analyze_with_context(prompt)
                else:
                    # Create a text-only analysis using the chart analysis method
                    return self.multimodal_llm.analyze_chart_image(None, prompt)
        except Exception as e:
            return {"analysis": f"LLM error: {str(e)}", "success": False}
    
    async def run_analysis(self, symbol: str, chart_path: str) -> Dict[str, Any]:
        """Run the complete agentic analysis"""
        initial_state = {
            "symbol": symbol,
            "chart_path": chart_path,
            "market_data": {},
            "technical_analysis": None,
            "fundamental_analysis": None,
            "news_analysis": None,
            "risk_analysis": None,
            "debate_history": [],
            "consensus": None,
            "final_recommendation": None,
            "current_round": 0,
            "max_rounds": 3,
            "messages": []
        }
        
        # Run the workflow
        result = await self.workflow.ainvoke(initial_state)
        
        return result["final_recommendation"]
    
    async def run_swarm_analysis(self, symbol: str, chart_path: str = None) -> Dict[str, Any]:
        """Run SwarmAI-style analysis with enhanced agent personalities"""
        
        print(f"🚀 Starting SwarmAI-LangGraph Analysis for {symbol}")
        print("=" * 60)
        
        # Gather market data first (CrewAI style - data-driven)
        market_data = await self._gather_market_data(symbol)
        
        # Initialize enhanced state with CrewAI features
        initial_state = {
            "symbol": symbol,
            "chart_path": chart_path,
            "market_data": market_data,
            "technical_analysis": None,
            "fundamental_analysis": None,
            "news_analysis": None,
            "risk_analysis": None,
            "research_analysis": None,
            "psychology_analysis": None,
            "trading_strategy": None,
            "debate_history": [],
            "consensus": None,
            "final_recommendation": None,
            "current_round": 0,
            "max_rounds": 3,
            "messages": [],
            "crew_mode": True,
            "agent_configs": self.crew_configs,
            "task_results": {}
        }
        
        # Execute CrewAI-style tasks sequentially
        results = []
        
        for i, task in enumerate(self.crew_tasks, 1):
            print(f"🤖 Agent {i}/6: {self.crew_configs[task.agent_role].role}")
            print(f"📋 Task: {task.description}")
            
            try:
                # Execute task with enhanced agent personality
                result = await self._execute_crew_task(task, initial_state)
                results.append(result)
                initial_state["task_results"][task.agent_role.value] = result
                
                print(f"✅ Completed: {result.get('summary', 'Task completed')[:100]}...")
                
            except Exception as e:
                print(f"❌ Task failed: {str(e)}")
                continue
            
            print()
        
        # Generate final crew report
        final_report = await self._generate_swarm_report(symbol, results, initial_state)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'SwarmAI-LangGraph',
            'individual_results': results,
            'final_report': final_report,
            'success': True
        }
    
    async def _execute_crew_task(self, task: CrewAITask, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CrewAI-style task with enhanced agent personality"""
        
        agent_config = self.crew_configs[task.agent_role]
        
        # Build CrewAI-style prompt
        prompt = self._build_crew_prompt(task, agent_config, state)
        
        # Execute based on agent type and available data
        if task.agent_role == AgentRole.TECHNICAL and state.get('chart_path'):
            # Use chart analysis if available
            response = self.multimodal_llm.chat_with_chart(state['chart_path'], prompt)
        else:
            # Use text-based analysis (standard CrewAI approach)
            response = self.multimodal_llm.analyze_with_context(prompt)
        
        # Parse and structure response
        parsed_result = self._parse_crew_response(response, task, agent_config)
        
        return parsed_result
    
    def _build_crew_prompt(self, task: CrewAITask, agent_config: CrewAIAgentConfig, state: Dict[str, Any]) -> str:
        """Build CrewAI-style prompt with agent personality"""
        
        # Get previous task results for context
        previous_results = self._format_previous_results(state.get('task_results', {}))
        
        prompt = f"""
        🎭 AGENT IDENTITY:
        Role: {agent_config.role}
        Goal: {agent_config.goal}
        
        📖 BACKSTORY:
        {agent_config.backstory}
        
        🎯 CURRENT TASK:
        {task.description}
        
        📋 EXPECTED OUTPUT:
        {task.expected_output}
        
        📊 CONTEXT:
        Symbol: {state.get('symbol', 'Unknown')}
        Market Data: {json.dumps(state.get('market_data', {}), indent=2, default=str)[:1000]}...
        
        💭 PREVIOUS TEAM INSIGHTS:
        {previous_results}
        
        🎨 RESPONSE FORMAT:
        As a {agent_config.role}, provide your analysis in this format:
        
        EXECUTIVE_SUMMARY: [3-sentence summary of your key findings]
        DETAILED_ANALYSIS: [Your comprehensive analysis in your expert domain]
        CONFIDENCE_SCORE: [0.0 to 1.0]
        KEY_INSIGHTS: [3-5 bullet points of actionable insights]
        RECOMMENDATIONS: [Specific recommendations from your expertise]
        RISKS_IDENTIFIED: [Key risks from your perspective]
        NEXT_STEPS: [What should be done next based on your analysis]
        
        🔍 REMEMBER:
        - Stay true to your role as {agent_config.role}
        - Focus on your area of expertise
        - Provide specific, actionable insights
        - Consider the team's previous findings
        - Be confident in your domain knowledge
        """
        
        return prompt
    
    def _format_previous_results(self, task_results: Dict[str, Any]) -> str:
        """Format previous task results for context"""
        if not task_results:
            return "No previous team insights available."
        
        formatted = []
        for agent_role, result in task_results.items():
            summary = result.get('summary', result.get('analysis', 'No summary available'))
            formatted.append(f"• {agent_role}: {summary[:150]}...")
        
        return "\n".join(formatted)
    
    def _parse_crew_response(self, response: Dict, task: CrewAITask, agent_config: CrewAIAgentConfig) -> Dict[str, Any]:
        """Parse CrewAI agent response"""
        
        if isinstance(response, dict):
            analysis_text = response.get("analysis", str(response))
        else:
            analysis_text = str(response)
        
        return {
            'agent_role': agent_config.role,
            'task_description': task.description,
            'analysis': analysis_text,
            'summary': self._extract_field(analysis_text, 'EXECUTIVE_SUMMARY'),
            'confidence': self._extract_confidence_score(analysis_text),
            'key_insights': self._extract_list_field(analysis_text, 'KEY_INSIGHTS'),
            'recommendations': self._extract_list_field(analysis_text, 'RECOMMENDATIONS'),
            'risks': self._extract_list_field(analysis_text, 'RISKS_IDENTIFIED'),
            'next_steps': self._extract_field(analysis_text, 'NEXT_STEPS'),
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a specific field from the response"""
        import re
        pattern = f'{field_name}:\\s*(.+?)(?=\\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score"""
        import re
        match = re.search(r'CONFIDENCE_SCORE:\s*([0-9.]+)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5
    
    def _extract_list_field(self, text: str, field_name: str) -> List[str]:
        """Extract a list field from the response"""
        import re
        pattern = f'{field_name}:\\s*(.+?)(?=\\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            items_text = match.group(1).strip()
            # Split by bullet points or numbers
            items = []
            for line in items_text.split('\n'):
                line = line.strip()
                if line:
                    # Remove common prefixes
                    line = re.sub(r'^[•\-\d+\.]\s*', '', line)
                    if line:
                        items.append(line)
            return items[:5]  # Limit to 5 items
        return []
    
    async def _generate_swarm_report(self, symbol: str, results: List[Dict], state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final SwarmAI swarm report"""
        
        # Aggregate metrics
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
        
        # Collect all insights and recommendations
        all_insights = []
        all_recommendations = []
        all_risks = []
        
        for result in results:
            all_insights.extend(result.get('key_insights', []))
            all_recommendations.extend(result.get('recommendations', []))
            all_risks.extend(result.get('risks', []))
        
        # Generate executive summary
        summary_prompt = f"""
        You are the Portfolio Manager synthesizing analysis from your expert team for {symbol}.

        SWARM ANALYSIS SUMMARY:
        {self._format_swarm_results_summary(results)}
        
        Create a comprehensive executive summary that:
        1. Highlights key consensus points across all experts
        2. Identifies areas of disagreement and how to resolve them
        3. Provides a unified investment thesis
        4. Includes specific action items with priorities
        5. Addresses key risks and mitigation strategies
        6. Provides clear next steps for decision makers
        
        Format as a professional investment committee report suitable for senior leadership.
        """
        
        try:
            summary_response = self.multimodal_llm.analyze_with_context(summary_prompt)
            executive_summary = summary_response.get('analysis', 'Summary generation failed')
        except Exception as e:
            executive_summary = f"Error generating summary: {str(e)}"
        
        return {
            'executive_summary': executive_summary,
            'overall_confidence': avg_confidence,
            'total_insights': len(all_insights),
            'key_insights': all_insights[:10],
            'total_recommendations': len(all_recommendations),
            'priority_recommendations': all_recommendations[:8],
            'risk_factors': all_risks[:6],
            'team_consensus': self._calculate_team_consensus(results),
            'investment_thesis': self._generate_investment_thesis(results),
            'action_items': self._generate_action_items(results),
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_swarm_results_summary(self, results: List[Dict]) -> str:
        """Format swarm results for executive summary"""
        formatted = []
        
        for result in results:
            formatted.append(f"""
            🤖 {result.get('agent_role', 'Unknown Agent')}:
            Summary: {result.get('summary', 'No summary')[:200]}...
            Confidence: {result.get('confidence', 0):.2f}
            Key Insight: {result.get('key_insights', ['None'])[0] if result.get('key_insights') else 'None'}
            Top Risk: {result.get('risks', ['None'])[0] if result.get('risks') else 'None'}
            """)
        
        return "\n".join(formatted)
    
    def _calculate_team_consensus(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate team consensus metrics"""
        if not results:
            return {'level': 'No consensus', 'score': 0.0}
        
        # Calculate based on confidence levels
        high_confidence = sum(1 for r in results if r.get('confidence', 0) > 0.7)
        consensus_score = high_confidence / len(results)
        
        if consensus_score >= 0.8:
            level = 'Strong consensus'
        elif consensus_score >= 0.6:
            level = 'Moderate consensus'
        else:
            level = 'Mixed opinions'
        
        return {
            'level': level,
            'score': consensus_score,
            'high_confidence_count': high_confidence,
            'total_agents': len(results)
        }
    
    def _generate_investment_thesis(self, results: List[Dict]) -> str:
        """Generate unified investment thesis"""
        # Extract key themes from all analyses
        themes = []
        for result in results:
            if result.get('summary'):
                themes.append(result['summary'])
        
        if themes:
            return f"Based on comprehensive analysis from {len(results)} specialists, the investment thesis centers on: " + "; ".join(themes[:3])
        return "Investment thesis pending complete analysis"
    
    def _generate_action_items(self, results: List[Dict]) -> List[Dict[str, str]]:
        """Generate prioritized action items"""
        action_items = []
        
        for result in results:
            next_steps = result.get('next_steps', '')
            if next_steps:
                action_items.append({
                    'action': next_steps[:100] + "..." if len(next_steps) > 100 else next_steps,
                    'owner': result.get('agent_role', 'Unknown'),
                    'priority': 'High' if result.get('confidence', 0) > 0.7 else 'Medium'
                })
        
        return action_items[:6]  # Top 6 action items
    
    async def _gather_market_data(self, symbol: str) -> Dict[str, Any]:
        """Gather comprehensive market data from multiple sources (CrewAI style) with a progress bar."""
        steps = [
            ("stock_data", "📈 Collecting stock data", self.data_tools.get_stock_data),
            ("news", "📰 Gathering financial news", self.data_tools.get_financial_news),
            ("technical_indicators", "📊 Calculating technical indicators", self.data_tools.calculate_technical_indicators),
            ("company_info", "🏢 Fetching company information", self.data_tools.get_company_info),
            ("sentiment", "📈 Analyzing market sentiment", self.data_tools.get_market_sentiment),
            ("support_resistance", "📊 Calculating support/resistance levels", self.data_tools.calculate_support_resistance),
            ("volume_analysis", "📈 Analyzing volume profile", self.data_tools.analyze_volume_profile),
        ]
        market_data: Dict[str, Any] = {}
        for key, msg, fn in tqdm(steps, desc=f"Market data collection for {symbol}", unit="task"):
            tqdm.write(f"{msg} for {symbol}...")
            try:
                result = fn(symbol)
                if asyncio.iscoroutine(result):
                    result = await result
                market_data[key] = result
            except Exception as e:
                market_data[key] = {"error": f"{msg.split()[1]} failed: {str(e)}"}
        tqdm.write("✅ Market data collection completed!")
        return market_data