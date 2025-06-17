import gradio as gr
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Import modules from existing files
from response_generator import generate_all_responses
from round_robin_evaluator import comprehensive_round_robin_evaluation, save_comprehensive_results
from realtime_detector import is_realtime_prompt
from search_fallback import get_google_snippets
from llm_prompt_eval_analysis import generate_visualizations, analyze_evaluation_data

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def check_api_keys():
    """Check if all required API keys are available."""
    keys_status = {}
    
    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    keys_status["OpenAI (GPT-4)"] = "Available" if openai_key else "Missing"
    
    # Check Claude
    claude_key = os.getenv("CLAUDE_API_KEY")
    keys_status["Claude 3"] = "Available" if claude_key else "Missing"
    
    # Check Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    keys_status["Gemini 1.5"] = "Available" if gemini_key else "Missing"
    
    # Check Google Search (optional)
    google_key = os.getenv("GOOGLE_API_KEY")
    google_cse = os.getenv("GOOGLE_CSE_ID")
    keys_status["Google Search"] = "Available" if (google_key and google_cse) else "Missing"
    
    return keys_status

def process_prompt(prompt, enable_realtime_detection, enable_evaluation, enable_analysis):
    """Process a prompt through the complete pipeline."""
    if not prompt.strip():
        return "Please enter a prompt.", None, None, None, None, None
    
    results = {
        "prompt": prompt,
        "responses": {},
        "evaluation": None,
        "analysis": None,
        "search_results": None,
        "is_realtime": False
    }
    
    # Step 1: Check if real-time detection is needed
    if enable_realtime_detection:
        try:
            results["is_realtime"] = is_realtime_prompt(prompt)
            if results["is_realtime"]:
                # Get Google search results
                search_results = get_google_snippets(prompt)
                results["search_results"] = search_results
                # Enhance prompt with search results
                enhanced_prompt = f"{prompt}\n\nRecent information: {search_results}"
            else:
                enhanced_prompt = prompt
        except Exception as e:
            print(f"Real-time detection error: {e}")
            enhanced_prompt = prompt
    else:
        enhanced_prompt = prompt
    
    # Step 2: Generate responses from all models
    try:
        responses = generate_all_responses(enhanced_prompt)
        results["responses"] = responses
    except Exception as e:
        return f"Error generating responses: {e}", None, None, None, None, None
    
    # Step 3: Perform evaluation if requested
    if enable_evaluation and responses:
        try:
            evaluation_results = comprehensive_round_robin_evaluation(responses, prompt)
            results["evaluation"] = evaluation_results
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = save_comprehensive_results(evaluation_results, prompt, timestamp)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
    
    # Step 4: Generate analysis if requested
    if enable_analysis and results["evaluation"]:
        try:
            # Create a temporary DataFrame for analysis
            analysis_data = []
            for model, data in results["evaluation"].items():
                for evaluator, eval_data in data.get('evaluations', {}).items():
                    row = {
                        'target_model': model,
                        'evaluator': evaluator,
                        'helpfulness': eval_data.get('helpfulness', 0.5),
                        'correctness': eval_data.get('correctness', 0.5),
                        'coherence': eval_data.get('coherence', 0.5),
                        'clarity': eval_data.get('clarity', 0.5),
                        'response': data.get('response', '')
                    }
                    analysis_data.append(row)
            
            if analysis_data:
                df = pd.DataFrame(analysis_data)
                results["analysis"] = df
                
        except Exception as e:
            print(f"Analysis error: {e}")
    
    return format_results(results)

def format_results(results):
    """Format results for Gradio display."""
    prompt = results["prompt"]
    responses = results["responses"]
    evaluation = results["evaluation"]
    analysis = results["analysis"]
    search_results = results["search_results"]
    is_realtime = results["is_realtime"]
    
    # Format responses
    responses_text = ""
    if responses:
        responses_text = "MODEL RESPONSES:\n" + "="*50 + "\n"
        for model, response in responses.items():
            responses_text += f"\n{model}:\n{'-'*20}\n{response}\n"
    else:
        responses_text = "No responses generated. Check API keys."
    
    # Format evaluation results
    evaluation_text = ""
    if evaluation:
        evaluation_text = "EVALUATION RESULTS:\n" + "="*50 + "\n"
        for model, data in evaluation.items():
            avg_scores = data.get('average_scores', {})
            evaluation_text += f"\n{model} Average Scores:\n"
            for metric, score in avg_scores.items():
                evaluation_text += f"  {metric}: {score}\n"
            evaluation_text += f"  Evaluated by: {list(data.get('evaluations', {}).keys())}\n"
    else:
        evaluation_text = "No evaluation performed."
    
    # Format search results
    search_text = ""
    if search_results and is_realtime:
        search_text = "REAL-TIME SEARCH RESULTS:\n" + "="*50 + "\n"
        search_text += search_results
    elif is_realtime:
        search_text = "Real-time query detected but search results unavailable."
    else:
        search_text = "Not a real-time query."
    
    # Create visualizations
    charts = []
    if analysis is not None and not analysis.empty:
        charts = create_visualizations(analysis)
    
    return responses_text, evaluation_text, search_text, charts

def create_visualizations(df):
    """Create Plotly visualizations for the analysis."""
    charts = []
    
    try:
        # 1. Model Performance Comparison
        if 'target_model' in df.columns:
            metrics = ['helpfulness', 'correctness', 'coherence', 'clarity']
            
            for metric in metrics:
                if metric in df.columns:
                    fig = px.box(df, x='target_model', y=metric, 
                               title=f'{metric.title()} Scores by Model',
                               color='target_model')
                    fig.update_layout(showlegend=False)
                    charts.append(fig)
        
        # 2. Evaluator Bias Analysis
        if 'evaluator' in df.columns:
            metrics = ['helpfulness', 'correctness', 'coherence', 'clarity']
            
            for metric in metrics:
                if metric in df.columns:
                    fig = px.box(df, x='evaluator', y=metric,
                               title=f'{metric.title()} Scores by Evaluator',
                               color='evaluator')
                    fig.update_layout(showlegend=False)
                    charts.append(fig)
        
        # 3. Heatmap of Cross-Evaluations
        if 'target_model' in df.columns and 'evaluator' in df.columns and 'helpfulness' in df.columns:
            pivot_data = df.pivot_table(
                values='helpfulness',
                index='target_model',
                columns='evaluator',
                aggfunc='mean'
            ).fillna(0)
            
            fig = px.imshow(pivot_data.values,
                           x=pivot_data.columns,
                           y=pivot_data.index,
                           title='Cross-Evaluation Heatmap (Helpfulness)',
                           color_continuous_scale='RdYlBu_r',
                           aspect='auto')
            fig.update_layout(xaxis_title='Evaluator', yaxis_title='Target Model')
            charts.append(fig)
    
    except Exception as e:
        print(f"Visualization error: {e}")
    
    return charts

def export_results(responses_text, evaluation_text, search_text):
    """Export results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/export_{timestamp}.txt"
    
    os.makedirs("results", exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("LLM COMPARISON RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(responses_text + "\n\n")
        f.write(evaluation_text + "\n\n")
        f.write(search_text + "\n\n")
    
    return f"Results exported to {filename}"

# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    # Check API keys
    api_status = check_api_keys()
    api_status_text = "API KEY STATUS:\n" + "="*30 + "\n"
    for service, status in api_status.items():
        api_status_text += f"{service}: {status}\n"
    
    with gr.Blocks(title="LLM Comparison Hub", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# LLM Comparison Hub")
        gr.Markdown("Compare responses from GPT-4, Claude 3, and Gemini 1.5 with comprehensive evaluation and analysis.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("## Input")
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Type your question or prompt here...",
                    lines=4
                )
                
                with gr.Row():
                    realtime_checkbox = gr.Checkbox(label="Enable real-time detection", value=True)
                    evaluation_checkbox = gr.Checkbox(label="Enable evaluation", value=True)
                    analysis_checkbox = gr.Checkbox(label="Enable analysis", value=True)
                
                process_btn = gr.Button("Process Prompt", variant="primary")
                
                # API status
                gr.Markdown("## API Status")
                api_status_display = gr.Textbox(
                    value=api_status_text,
                    label="API Keys",
                    lines=len(api_status) + 3,
                    interactive=False
                )
            
            with gr.Column(scale=3):
                # Output section
                gr.Markdown("## Results")
                
                with gr.Tabs():
                    with gr.TabItem("Responses"):
                        responses_output = gr.Textbox(
                            label="Model Responses",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.TabItem("Evaluation"):
                        evaluation_output = gr.Textbox(
                            label="Evaluation Results",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.TabItem("Search Results"):
                        search_output = gr.Textbox(
                            label="Real-time Search Results",
                            lines=10,
                            interactive=False
                        )
                    
                    with gr.TabItem("Visualizations"):
                        charts_output = gr.Plot(label="Analysis Charts")
                
                # Export button
                export_btn = gr.Button("Export Results")
                export_output = gr.Textbox(label="Export Status", interactive=False)
        
        # Event handlers
        process_btn.click(
            fn=process_prompt,
            inputs=[prompt_input, realtime_checkbox, evaluation_checkbox, analysis_checkbox],
            outputs=[responses_output, evaluation_output, search_output, charts_output]
        )
        
        export_btn.click(
            fn=export_results,
            inputs=[responses_output, evaluation_output, search_output],
            outputs=[export_output]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 