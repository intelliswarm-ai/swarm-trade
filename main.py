import os
import sys
import logging
from datetime import datetime
from src.screenshot_capture import ScreenshotCapture
from src.enhanced_image_analyzer import EnhancedImageAnalyzer
from src.enhanced_news_analyzer import EnhancedNewsAnalyzer
from src.signal_generator import SignalGenerator
from src.console_display import ConsoleDisplay
from src.multimodal_llm import MultimodalLLM
from src.langgraph_agents import LangGraphAgentWorkflow
from src.agent_tools import AgentTools
from src.command_efficiency_evaluator import evaluator, measure_execution
from src.cache_manager import get_cache_status, clear_all_caches, cleanup_all_caches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_signals.log'),
        logging.StreamHandler()
    ]
)

class ForexSignalApp:
    def __init__(self, use_gui=False):
        self.screenshot_capture = ScreenshotCapture()
        self.image_analyzer = EnhancedImageAnalyzer()
        self.news_analyzer = EnhancedNewsAnalyzer()
        self.signal_generator = SignalGenerator()
        self.console_display = ConsoleDisplay()
        self.multimodal_llm = MultimodalLLM()
        self.agent_tools = AgentTools()
        self.agent_workflow = LangGraphAgentWorkflow(self.multimodal_llm, self.agent_tools)
        self.use_gui = use_gui
        
    def run_analysis(self):
        try:
            logging.info("Starting forex signal analysis...")
            
            screenshot_path = self.screenshot_capture.capture_trading_screen()
            if not screenshot_path:
                logging.error("Failed to capture screenshot")
                print("❌ Failed to capture screenshot. Please ensure a trading application is running.")
                return
            
            from pathlib import Path
            print(f"✅ Screenshot captured: {Path(screenshot_path).as_posix()}")
            
            chart_data = self.image_analyzer.analyze_chart(screenshot_path)
            news_sentiment = self.news_analyzer.get_forex_sentiment()
            
            # Show analysis summary
            self.console_display.show_analysis_summary(chart_data, news_sentiment)
            
            signals = self.signal_generator.generate_signals(chart_data, news_sentiment)
            
            # Display signals
            self.console_display.show_signals(signals)
            
            # Export signals
            if signals:
                self.console_display.export_signals_to_file(signals)
            
            # LLM Analysis (Quick analysis for faster results)
            print("\n🤖 Starting Quick LLM Analysis...")
            llm_analysis = self.multimodal_llm.quick_analysis(screenshot_path)
            
            if llm_analysis.get('success'):
                print("\n🧠 LLM Analysis Results:")
                print("=" * 60)
                print(llm_analysis['analysis'])
                print("=" * 60)
                print(f"Model: {llm_analysis['model']}")
            else:
                print(f"⚠️  LLM Analysis failed: {llm_analysis.get('message', 'Unknown error')}")
                if 'suggestion' in llm_analysis:
                    print(f"💡 Suggestion: {llm_analysis['suggestion']}")
            
            logging.info("Analysis completed successfully")
            
        except Exception as e:
            logging.error(f"Error in analysis: {str(e)}")
            print(f"❌ Error in analysis: {str(e)}")
    
    def run_gui_mode(self):
        try:
            from src.ui_display import UIDisplay
            ui = UIDisplay()
            ui.run()
        except Exception as e:
            print(f"❌ Error starting GUI: {str(e)}")
            print("Try running in console mode instead: python main.py")
    
    def run_chat_mode(self):
        """Interactive chat mode with LLM"""
        print("🤖 Interactive Chart Analysis Chat Mode")
        print("=" * 50)
        print("Commands:")
        print("  /analyze [path] - Analyze a chart image")
        print("  /screenshot - Take screenshot and analyze")
        print("  /quick - Get quick analysis (faster)")
        print("  /signals - Get trading signals from LLM")
        print("  /patterns - Get chart pattern analysis")
        print("  /risk - Get risk assessment")
        print("  /sentiment - Get market sentiment analysis")
        print("  /summary - Get comprehensive analysis")
        print("  /agents [symbol] - Run agentic workflow analysis (no chart required)")
        print("  /swarm [symbol] - Run SwarmAI-style team analysis (no chart required)")
        print("  /debate [symbol] - Start agent debate for symbol")
        print("  /consensus - Get consensus from agent analysis")
        print("  /tools - Show available agent tools")
        print("  /tool [tool_name] [args] - Execute specific tool")
        print("  /history - Show chat history")
        print("  /clear - Clear chat history")
        print("  /models - Show available models")
        print("  /switch [model] - Switch to different model")
        print("  /timeout [seconds] - Set timeout (default: 300s)")
        print("  /efficiency - Show command efficiency report")
        print("  /metrics - Export efficiency metrics to file")
        print("  /benchmark - Run command benchmarks")
        print("  /cache - Show cache status and statistics")
        print("  /clearcache - Clear all caches to free memory")
        print("  /help - Show this help")
        print("  /exit - Exit chat mode")
        print("=" * 50)
        
        current_image = None
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['/exit', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == '/help':
                    print("Available commands listed above")
                    
                elif user_input.lower() == '/screenshot':
                    print("📸 Taking screenshot...")
                    with measure_execution("screenshot_capture"):
                        screenshot_path = self.screenshot_capture.capture_trading_screen()
                    if screenshot_path:
                        current_image = screenshot_path
                        from pathlib import Path
                        print(f"✅ Screenshot saved: {Path(screenshot_path).as_posix()}")

                        # Auto-analyze the screenshot
                        print("🤖 Analyzing screenshot...")
                        with measure_execution("screenshot_analysis"):
                            result = self.multimodal_llm.analyze_chart_image(screenshot_path)
                        self.display_llm_result(result)
                    else:
                        print("❌ Failed to take screenshot")
                
                elif user_input.startswith('/analyze'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        image_path = parts[1]
                        if os.path.exists(image_path):
                            current_image = image_path
                            with measure_execution("image_analysis"):
                                result = self.multimodal_llm.analyze_chart_image(image_path)
                            self.display_llm_result(result)
                        else:
                            print(f"❌ Image not found: {image_path}")
                    else:
                        print("❌ Please provide image path: /analyze <path>")
                
                elif user_input.lower() == '/quick':
                    if current_image:
                        with measure_execution("quick_analysis"):
                            result = self.multimodal_llm.quick_analysis(current_image)
                        self.display_llm_result(result)
                    else:
                        print("❌ No image loaded. Use /screenshot or /analyze first")
                
                elif user_input.lower() == '/signals':
                    if current_image:
                        result = self.multimodal_llm.get_trading_signals_from_llm(current_image)
                        self.display_llm_result(result)
                    else:
                        print("❌ No image loaded. Use /screenshot or /analyze first")
                
                elif user_input.lower() == '/patterns':
                    if current_image:
                        result = self.multimodal_llm.get_chart_pattern_analysis(current_image)
                        self.display_llm_result(result)
                    else:
                        print("❌ No image loaded. Use /screenshot or /analyze first")
                
                elif user_input.lower() == '/risk':
                    if current_image:
                        result = self.multimodal_llm.get_risk_assessment(current_image)
                        self.display_llm_result(result)
                    else:
                        print("❌ No image loaded. Use /screenshot or /analyze first")
                
                elif user_input.lower() == '/sentiment':
                    if current_image:
                        result = self.multimodal_llm.get_market_sentiment_analysis(current_image)
                        self.display_llm_result(result)
                    else:
                        print("❌ No image loaded. Use /screenshot or /analyze first")
                
                elif user_input.lower() == '/summary':
                    if current_image:
                        result = self.multimodal_llm.summarize_analysis(current_image)
                        self.display_llm_result(result)
                    else:
                        print("❌ No image loaded. Use /screenshot or /analyze first")
                
                elif user_input.lower() == '/history':
                    history = self.multimodal_llm.get_chat_history()
                    if history:
                        print("\n📜 Chat History:")
                        for i, entry in enumerate(history[-5:], 1):  # Show last 5
                            print(f"{i}. Q: {entry['question'][:100]}...")
                            print(f"   A: {entry['response'][:200]}...")
                            print()
                    else:
                        print("📜 No chat history")
                
                elif user_input.lower() == '/clear':
                    self.multimodal_llm.clear_chat_history()
                
                elif user_input.lower() == '/models':
                    models = self.multimodal_llm.get_available_models()
                    print(f"🤖 Available models: {models}")
                    print(f"🔸 Current model: {self.multimodal_llm.model}")
                
                elif user_input.startswith('/switch'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        model_name = parts[1]
                        self.multimodal_llm.switch_model(model_name)
                    else:
                        print("❌ Please provide model name: /switch <model>")
                
                elif user_input.startswith('/timeout'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        try:
                            timeout_seconds = int(parts[1])
                            if timeout_seconds > 0:
                                self.multimodal_llm.set_timeout(timeout_seconds)
                            else:
                                print("❌ Timeout must be positive")
                        except ValueError:
                            print("❌ Please provide a valid number: /timeout <seconds>")
                    else:
                        current_timeout = self.multimodal_llm.get_timeout()
                        print(f"🔸 Current timeout: {current_timeout} seconds ({current_timeout//60} minutes)")
                        print("💡 Usage: /timeout <seconds> (e.g., /timeout 600 for 10 minutes)")
                
                elif user_input.startswith('/agents'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        symbol = parts[1].upper().replace('$', '')
                        print(f"🤖 Starting agentic workflow analysis for {symbol}...")
                        print("🔄 Deploying specialized agents...")
                        try:
                            import asyncio
                            chart_path = current_image if current_image else None
                            with measure_execution("agentic_workflow"):
                                result = asyncio.run(self.agent_workflow.run_analysis(symbol, chart_path))
                            self.display_agent_result(result, symbol)
                        except Exception as e:
                            print(f"❌ Error in agentic analysis: {str(e)}")
                            print(f"💡 Debug: Available methods: {[m for m in dir(self.multimodal_llm) if not m.startswith('_')]}" )
                    else:
                        print("❌ Please provide symbol: /agents <symbol>")
                
                elif user_input.startswith('/crew'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        symbol = parts[1].upper().replace('$', '')  # Remove $ prefix if present
                        print(f"🚀 Starting SwarmAI-style team analysis for {symbol}...")
                        print("👥 Deploying specialized expert team...")
                        print("📊 Gathering market data from multiple sources...")
                        try:
                            import asyncio
                            # Use chart if available, but not required for CrewAI analysis
                            chart_path = current_image if current_image else None
                            result = asyncio.run(self.agent_workflow.run_swarm_analysis(symbol, chart_path))
                            self.display_swarm_result(result, symbol)
                        except Exception as e:
                            print(f"❌ Error in CrewAI analysis: {str(e)}")
                            print(f"💡 Debug: Available methods: {[method for method in dir(self.multimodal_llm) if not method.startswith('_')]}")
                    else:
                        print("❌ Please provide symbol: /swarm <symbol>")
                
                elif user_input.startswith('/debate'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        symbol = parts[1].upper().replace('$', '')  # Remove $ prefix if present
                        if current_image:
                            print(f"🗣️  Starting agent debate for {symbol}...")
                            print("🤖 Agents are analyzing and debating...")
                            try:
                                import asyncio
                                result = asyncio.run(self.agent_workflow.run_analysis(symbol, current_image))
                                self.display_debate_result(result, symbol)
                            except Exception as e:
                                print(f"❌ Error in agent debate: {str(e)}")
                                print(f"💡 Debug: Available methods: {[method for method in dir(self.multimodal_llm) if not method.startswith('_')]}")
                        else:
                            print("❌ No image loaded. Use /screenshot or /analyze first")
                    else:
                        print("❌ Please provide symbol: /debate <symbol>")
                
                elif user_input.lower() == '/tools':
                    print("🛠️  Available Agent Tools:")
                    print("=" * 40)
                    tools = self.agent_tools.get_tool_descriptions()
                    for tool_name, description in tools.items():
                        print(f"  {tool_name}: {description}")
                    print("=" * 40)
                    print("💡 Usage: /tool <tool_name> <args>")
                
                elif user_input.startswith('/tool'):
                    parts = user_input.split(' ', 2)
                    if len(parts) >= 2:
                        tool_name = parts[1]
                        args = parts[2] if len(parts) > 2 else ""
                        
                        print(f"🛠️  Executing tool: {tool_name}")
                        try:
                            # Parse arguments (simplified)
                            if args:
                                # Try to parse as key=value pairs
                                kwargs = {}
                                if '=' in args:
                                    for arg in args.split():
                                        if '=' in arg:
                                            key, value = arg.split('=', 1)
                                            kwargs[key] = value
                                else:
                                    # Single argument, assume it's symbol
                                    kwargs['symbol'] = args
                                
                                with measure_execution(f"tool_{tool_name}"):
                                    result = self.agent_tools.execute_tool(tool_name, **kwargs)
                            else:
                                with measure_execution(f"tool_{tool_name}"):
                                    result = self.agent_tools.execute_tool(tool_name)
                            
                            self.display_tool_result(result, tool_name)
                        except Exception as e:
                            print(f"❌ Error executing tool: {str(e)}")
                    else:
                        print("❌ Please provide tool name: /tool <tool_name> [args]")
                
                elif user_input.lower() == '/efficiency':
                    print("📊 Generating efficiency report...")
                    report = evaluator.get_efficiency_report()
                    self.display_efficiency_report(report)
                
                elif user_input.startswith('/metrics'):
                    parts = user_input.split()
                    filename = parts[1] if len(parts) > 1 else "command_metrics.json"
                    print(f"💾 Exporting metrics to {filename}...")
                    evaluator.export_metrics(filename)
                    print(f"✅ Metrics exported to {filename}")
                
                elif user_input.lower() == '/benchmark':
                    print("🏁 Running command benchmarks...")
                    print("This will test various commands for performance analysis...")
                    self.run_benchmark_tests()
                
                elif user_input.lower() == '/cache':
                    print("💾 Cache Status and Statistics")
                    print("=" * 40)
                    cache_status = get_cache_status()
                    for cache_name, stats in cache_status.items():
                        print(f"\n{cache_name.replace('_', ' ').title()}:")
                        for key, value in stats.items():
                            print(f"  {key.replace('_', ' ').title()}: {value}")
                
                elif user_input.lower() == '/clearcache':
                    print("🧹 Clearing all caches...")
                    clear_all_caches()
                    cleanup_stats = cleanup_all_caches()
                    print("✅ All caches cleared successfully!")
                    print(f"📊 Cleaned up: {sum(cleanup_stats.values())} expired entries")
                
                elif user_input.strip() and not user_input.startswith('/'):
                    # Regular chat about the current image
                    if current_image:
                        print("🤖 Analyzing your question...")
                        result = self.multimodal_llm.chat_with_chart(current_image, user_input)
                        self.display_llm_result(result)
                    else:
                        print("❌ No image loaded. Use /screenshot or /analyze first")
                
                else:
                    print("❌ Unknown command. Type /help for available commands")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def display_llm_result(self, result):
        """Display LLM result in a formatted way"""
        if result.get('success'):
            print("\n🤖 LLM Response:")
            print("=" * 60)
            print(result['analysis'])
            print("=" * 60)
            print(f"Model: {result['model']}")
        else:
            print(f"⚠️  LLM failed: {result.get('message', 'Unknown error')}")
            if 'suggestion' in result:
                print(f"💡 Suggestion: {result['suggestion']}")
    
    def display_efficiency_report(self, report):
        """Display efficiency evaluation report"""
        print("\n📊 COMMAND EFFICIENCY REPORT")
        print("=" * 60)
        
        if "summary" in report:
            print("\n📋 SUMMARY:")
            for key, value in report["summary"].items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        
        if "command_analytics" in report and report["command_analytics"]:
            print("\n📈 COMMAND PERFORMANCE:")
            for cmd_name, metrics in list(report["command_analytics"].items())[:5]:
                print(f"\n   {cmd_name}:")
                print(f"      Success Rate: {metrics['success_rate']}")
                print(f"      Avg Time: {metrics['avg_execution_time']}")
                print(f"      Executions: {metrics['total_executions']}")
        
        if "performance_rankings" in report:
            rankings = report["performance_rankings"]
            if rankings.get("fastest_commands"):
                print(f"\n🏆 FASTEST COMMANDS:")
                for cmd in rankings["fastest_commands"][:3]:
                    print(f"   • {cmd}")
        
        if "recommendations" in report:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"][:3]:
                print(f"   • {rec}")
        
        print("\n" + "=" * 60)
    
    def run_benchmark_tests(self):
        """Run benchmark tests for command evaluation"""
        print("🔄 Running benchmark tests...")
        
        # Test various commands to generate metrics
        test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        if hasattr(self, 'agent_tools'):
            print("📊 Testing data fetching tools...")
            for symbol in test_symbols:
                try:
                    with measure_execution(f"benchmark_stock_data"):
                        self.agent_tools.execute_tool("get_stock_data", symbol=symbol, period="1mo")
                    print(f"   ✅ {symbol} data fetch completed")
                except Exception as e:
                    print(f"   ❌ {symbol} failed: {str(e)}")
            
            print("\n🔧 Testing analysis tools...")
            analysis_tools = ["get_company_info", "calculate_technical_indicators", "get_financial_news"]
            for tool in analysis_tools:
                try:
                    with measure_execution(f"benchmark_{tool}"):
                        if tool == "calculate_technical_indicators":
                            self.agent_tools.execute_tool(tool, symbol="AAPL", indicators=["sma_20", "rsi"])
                        else:
                            self.agent_tools.execute_tool(tool, symbol="AAPL")
                    print(f"   ✅ {tool} completed")
                except Exception as e:
                    print(f"   ❌ {tool} failed: {str(e)}")
        
        print("\n✅ Benchmark tests completed!")

    def display_agent_result(self, result, symbol):
        """Display agentic workflow result"""
        if not result:
            print("❌ No result from agentic workflow")
            return
        
        print(f"\n🤖 Agentic Analysis Results for {symbol}")
        print("=" * 70)
        
        # Final recommendation
        print(f"📊 Final Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
        
        # Individual agent opinions
        agent_opinions = result.get('agent_opinions', {})
        if agent_opinions:
            print("\n👥 Agent Opinions:")
            print("-" * 50)
            
            for agent_type, opinion in agent_opinions.items():
                if opinion:
                    print(f"\n🔸 {agent_type.title().replace('_', ' ')} Agent:")
                    print(f"   Recommendation: {opinion.get('recommendation', 'N/A')}")
                    print(f"   Confidence: {opinion.get('confidence', 0):.2f}")
                    print(f"   Reasoning: {opinion.get('reasoning', 'N/A')[:200]}...")
        
        # Debate history
        debate_history = result.get('debate_history', [])
        if debate_history:
            print(f"\n💬 Debate Rounds: {len(debate_history)}")
            for i, round_info in enumerate(debate_history, 1):
                print(f"   Round {i}: {round_info.get('topic', 'N/A')}")
        
        # Final synthesis
        synthesis = result.get('synthesis', '')
        if synthesis:
            print(f"\n📝 Final Analysis:")
            print("-" * 50)
            print(synthesis[:500] + "..." if len(synthesis) > 500 else synthesis)
        
        print("=" * 70)
    
    def display_debate_result(self, result, symbol):
        """Display agent debate result with focus on the debate process"""
        if not result:
            print("❌ No result from agent debate")
            return
        
        print(f"\n🗣️  Agent Debate Results for {symbol}")
        print("=" * 70)
        
        # Show each agent's initial position
        agent_opinions = result.get('agent_opinions', {})
        if agent_opinions:
            print("🎭 Initial Agent Positions:")
            print("-" * 50)
            
            for agent_type, opinion in agent_opinions.items():
                if opinion:
                    print(f"\n🤖 {agent_type.title().replace('_', ' ')} Agent:")
                    print(f"   Position: {opinion.get('recommendation', 'N/A')}")
                    print(f"   Confidence: {opinion.get('confidence', 0):.2f}")
                    print(f"   Key Argument: {opinion.get('reasoning', 'N/A')[:150]}...")
        
        # Show debate rounds
        debate_history = result.get('debate_history', [])
        if debate_history:
            print(f"\n💬 Debate Progression ({len(debate_history)} rounds):")
            print("-" * 50)
            
            for i, round_info in enumerate(debate_history, 1):
                print(f"\n🔄 Round {i}: {round_info.get('topic', 'N/A')}")
                if 'opinions' in round_info:
                    for opinion in round_info['opinions']:
                        agent_name = opinion[0] if isinstance(opinion, tuple) else opinion.get('agent_id', 'Unknown')
                        print(f"   {agent_name}: {opinion[1].get('recommendation', 'N/A') if isinstance(opinion, tuple) else opinion.get('recommendation', 'N/A')}")
        
        # Final consensus
        print(f"\n🎯 Final Consensus:")
        print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        
        print("=" * 70)
    
    def display_swarm_result(self, result, symbol):
        """Display SwarmAI-style swarm analysis result"""
        if not result:
            print("❌ No result from CrewAI analysis")
            return
        
        print(f"\n🚀 SwarmAI Expert Team Analysis for {symbol}")
        print("=" * 70)
        
        # Final report summary
        final_report = result.get('final_report', {})
        if final_report:
            print(f"📊 Overall Confidence: {final_report.get('overall_confidence', 0):.2f}")
            print(f"🎯 Team Consensus: {final_report.get('team_consensus', {}).get('level', 'Unknown')}")
            print(f"📈 Investment Thesis: {final_report.get('investment_thesis', 'N/A')}")
            
            # Executive summary
            executive_summary = final_report.get('executive_summary', '')
            if executive_summary:
                print(f"\n📋 Executive Summary:")
                print("-" * 50)
                print(executive_summary[:800] + "..." if len(executive_summary) > 800 else executive_summary)
            
            # Key insights
            key_insights = final_report.get('key_insights', [])
            if key_insights:
                print(f"\n💡 Key Insights:")
                print("-" * 50)
                for i, insight in enumerate(key_insights[:5], 1):
                    print(f"{i}. {insight}")
            
            # Priority recommendations
            recommendations = final_report.get('priority_recommendations', [])
            if recommendations:
                print(f"\n🎯 Priority Recommendations:")
                print("-" * 50)
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"{i}. {rec}")
            
            # Risk factors
            risks = final_report.get('risk_factors', [])
            if risks:
                print(f"\n⚠️  Risk Factors:")
                print("-" * 50)
                for i, risk in enumerate(risks[:4], 1):
                    print(f"{i}. {risk}")
            
            # Action items
            action_items = final_report.get('action_items', [])
            if action_items:
                print(f"\n📋 Action Items:")
                print("-" * 50)
                for item in action_items[:4]:
                    print(f"• {item.get('action', 'N/A')} [{item.get('priority', 'Medium')} Priority]")
                    print(f"  Owner: {item.get('owner', 'Unknown')}")
        
        # Individual expert results
        individual_results = result.get('individual_results', [])
        if individual_results:
            print(f"\n👥 Individual Expert Analysis:")
            print("-" * 50)
            
            for expert in individual_results:
                print(f"\n🤖 {expert.get('agent_role', 'Unknown Expert')}:")
                print(f"   Summary: {expert.get('summary', 'No summary')[:150]}...")
                print(f"   Confidence: {expert.get('confidence', 0):.2f}")
                if expert.get('key_insights'):
                    print(f"   Key Insight: {expert.get('key_insights', ['None'])[0]}")
        
        print("=" * 70)
    
    def display_tool_result(self, result, tool_name):
        """Display tool execution result"""
        print(f"\n🛠️  Tool Result: {tool_name}")
        print("=" * 50)
        
        if isinstance(result, dict):
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                # Pretty print the result
                import json
                try:
                    print(json.dumps(result, indent=2, default=str))
                except:
                    print(str(result))
        else:
            print(str(result))
        
        print("=" * 50)
    
    def start_continuous_monitoring(self):
        import schedule
        import time
        
        schedule.every(5).minutes.do(self.run_analysis)
        
        logging.info("Starting continuous monitoring...")
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    print("🚀 Swarm Trade - Forex Signal Analyzer")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--gui":
            print("Starting GUI mode...")
            app = ForexSignalApp(use_gui=True)
            app.run_gui_mode()
        elif sys.argv[1] == "--continuous":
            print("Starting continuous monitoring mode...")
            app = ForexSignalApp()
            app.start_continuous_monitoring()
        elif sys.argv[1] == "--chat":
            print("Starting interactive chat mode...")
            app = ForexSignalApp()
            app.run_chat_mode()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python main.py              - Single analysis (console)")
            print("  python main.py --gui        - GUI mode")
            print("  python main.py --chat       - Interactive chat mode with LLM")
            print("  python main.py --continuous - Continuous monitoring")
            print("  python main.py --help       - Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        print("Starting single analysis mode...")
        app = ForexSignalApp()
        app.run_analysis()