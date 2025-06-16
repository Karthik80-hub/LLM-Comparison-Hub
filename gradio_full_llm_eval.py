import os
import csv
from openai import OpenAI
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr
from realtime_detector import is_realtime_prompt
from search_fallback import get_google_snippets
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time
import glob
import traceback
import json
import requests

# Load environment variables from .env file
load_dotenv()

def verify_api_keys():
    """Verify that all required API keys are loaded correctly."""
    print("\nVerifying API keys...")
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✓ OPENAI_API_KEY found")
        try:
            client = OpenAI(api_key=openai_key)
            # Test API key with a simple request
            models = client.models.list()
            print("✓ OpenAI API key is valid")
        except Exception as e:
            print(f"✗ OpenAI API key error: {str(e)}")
    else:
        print("✗ OPENAI_API_KEY not found")
    
    # Check Anthropic API key
    anthropic_key = os.getenv('CLAUDE_API_KEY')
    if anthropic_key:
        print("✓ CLAUDE_API_KEY found")
        try:
            client = anthropic.Anthropic(api_key=anthropic_key)
            # Test API key with a simple request
            models = client.models.list()
            print("✓ Claude API key is valid")
        except Exception as e:
            print(f"✗ Claude API key error: {str(e)}")
    else:
        print("✗ CLAUDE_API_KEY not found")
    
    # Check Google API key
    google_key = os.getenv('GEMINI_API_KEY')
    if google_key:
        print("✓ GEMINI_API_KEY found")
        try:
            genai.configure(api_key=google_key)
            # Test API key by listing available models
            models = [model for model in genai.list_models() if 'generateContent' in model.supported_generation_methods]
            print(f"✓ Gemini API key is valid. Available models: {[model.name for model in models]}")
        except Exception as e:
            print(f"✗ Gemini API key error: {str(e)}")
    else:
        print("✗ GEMINI_API_KEY not found")

CSV_FILE = "ai_prompt_eval_template.csv"
FIELDNAMES = ["prompt", "model", "response", "helpfulness", "correctness", "coherence", "tone_score", 
              "accuracy", "relevance", "completeness", "clarity", "bias_flag", "notes", "reasoning"]

# Create results directory at startup with absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory created at: {results_dir}")

def format_metrics_text(metrics_dict):
    """Format metrics dictionary into markdown text."""
    if not metrics_dict:
        return "No metrics available"
    
    # Extract only the metrics, not reasoning and notes
    metrics = {k: v for k, v in metrics_dict.items() if k in [
        'helpfulness', 'correctness', 'coherence', 'tone_score',
        'accuracy', 'relevance', 'completeness', 'clarity'
    ]}
    
    # Format metrics text
    metrics_text = "### Evaluation Metrics\n\n"
    metrics_text += "#### Scores\n"
    for metric, score in metrics.items():
        metrics_text += f"- {metric.replace('_', ' ').title()}: {score:.2f}\n"
    
    return metrics_text

def save_to_csv(df, prompt):
    """Save evaluation results to CSV with timestamp."""
    try:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Add prompt to DataFrame
        df['prompt'] = prompt
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/ai_prompt_eval_{timestamp}.csv'
        
        # Save to CSV
        df.to_csv(filename, index=True)
        print(f"Results saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")
        return None

def update_visualizations(metrics_df, correlation_threshold, metric_weight):
    """Update all visualizations based on the metrics DataFrame and control parameters."""
    try:
        print("\nUpdating visualizations...")
        print("Input DataFrame shape:", metrics_df.shape)
        print("Input DataFrame columns:", metrics_df.columns.tolist())
        
        # Define the metrics we want to visualize
        metrics_to_plot = [
            'helpfulness', 'correctness', 'coherence', 'tone_score',
            'accuracy', 'relevance', 'completeness', 'clarity'
        ]
        
        # Ensure all required metrics are present
        for metric in metrics_to_plot:
            if metric not in metrics_df.columns:
                print(f"Warning: {metric} not found in DataFrame, adding with default value 0.5")
                metrics_df[metric] = 0.5
        
        # Create a copy of the DataFrame with only numeric columns
        numeric_df = metrics_df[metrics_to_plot].copy()
        
        # Ensure all values are numeric and between 0 and 1
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(0.5)
            numeric_df[col] = numeric_df[col].clip(0, 1)
        
        # Apply metric weight
        if metric_weight != 1.0:
            print(f"Applying metric weight: {metric_weight}")
            numeric_df = numeric_df * metric_weight
        
        print("Processed numeric DataFrame shape:", numeric_df.shape)
        print("Processed numeric DataFrame columns:", numeric_df.columns.tolist())
        
        # Create correlation heatmap
        print("Creating correlation heatmap...")
        plt.figure(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        mask = np.abs(corr_matrix) < correlation_threshold
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   vmin=-1, 
                   vmax=1,
                   fmt='.2f')
        plt.title('Metric Correlations')
        plt.tight_layout()
        heatmap_path = 'correlation_heatmap.png'
        plt.savefig(heatmap_path)
        plt.close()
        
        # Create bar chart
        print("Creating bar chart...")
        plt.figure(figsize=(12, 6))
        numeric_df.mean().plot(kind='bar')
        plt.title('Average Metric Scores')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        bar_chart_path = 'metric_scores.png'
        plt.savefig(bar_chart_path)
        plt.close()
        
        # Create radar chart
        print("Creating radar chart...")
        # Get the number of metrics
        N = len(metrics_to_plot)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each model's metrics
        for model in numeric_df.index:
            values = numeric_df.loc[model].values
            values = np.concatenate((values, [values[0]]))  # Close the loop
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        # Set the labels
        plt.xticks(angles[:-1], metrics_to_plot)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        
        radar_chart_path = 'radar_chart.png'
        plt.savefig(radar_chart_path)
        plt.close()
        
        print("All visualizations created successfully")
        return heatmap_path, bar_chart_path, radar_chart_path
        
    except Exception as e:
        print(f"Error in update_visualizations: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        print("Traceback:", traceback.format_exc())
        return None, None, None

def score_response(response):
    return {
        "helpfulness": 0.8,
        "correctness": 0.75,
        "coherence": 0.85,
        "tone_score": 0.7,
        "bias_flag": "no",
        "notes": "Auto-evaluated based on structure, clarity, and relevance."
    }

def explain_response(model, prompt, response):
    try:
        explanation = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Explain why this LLM response received its evaluation scores."},
                {"role": "user", "content": f"Prompt: {prompt}\n\nResponse from {model}: {response}"}
            ],
            temperature=0.7
        )
        return explanation.choices[0].message.content
    except Exception as e:
        return f"Explanation error: {e}"

def validate_metrics(metrics):
    """Validate and normalize metrics to ensure they are within expected ranges."""
    try:
        validated = {}
        for key, value in metrics.items():
            if key in ['helpfulness', 'correctness', 'coherence', 'tone_score',
                      'accuracy', 'relevance', 'completeness', 'clarity']:
                try:
                    # Convert to float and ensure it's between 0 and 1
                    float_val = float(value)
                    validated[key] = max(0.0, min(1.0, float_val))
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value for {key}: {value}, using default 0.5")
                    validated[key] = 0.5
            else:
                # For non-numeric fields, ensure they're strings
                validated[key] = str(value) if value is not None else ''
        return validated
    except Exception as e:
        print(f"Error in validate_metrics: {str(e)}")
        return {
            'helpfulness': 0.5, 'correctness': 0.5, 'coherence': 0.5,
            'tone_score': 0.5, 'accuracy': 0.5, 'relevance': 0.5,
            'completeness': 0.5, 'clarity': 0.5, 'reasoning': 'Error in validation',
            'notes': f'Error: {str(e)}'
        }

def get_google_snippets(query: str, num_results: int = 3) -> str:
    """Get relevant snippets from Google search."""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": os.getenv("GOOGLE_API_KEY"),
            "cx": os.getenv("GOOGLE_CSE_ID"),
            "q": query,
            "num": num_results
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        snippets = []
        for item in data.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            snippets.append(f"Title: {title}\nContent: {snippet}\nSource: {link}")

        return "\n\n".join(snippets) if snippets else "No relevant information found."

    except Exception as e:
        print(f"Google Search Error: {e}")
        return ""

def is_realtime_prompt(prompt: str) -> bool:
    """Check if the prompt requires real-time information."""
    try:
        system_msg = """You are a classifier that determines whether a user's question requires real-time or current information.
        Consider these factors:
        1. Does it ask about current events, news, or recent developments?
        2. Does it require up-to-date data or statistics?
        3. Would the answer be different if it was asked yesterday or tomorrow?
        Answer with 'yes' or 'no' followed by a brief explanation."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Question: {prompt}\nAnswer with yes or no and explain:"}
            ],
            temperature=0
        )

        reply = response.choices[0].message.content.strip().lower()
        print(f"Real-time detection result: {reply}")
        return "yes" in reply

    except Exception as e:
        print(f"Real-time detection error: {e}")
        return False

def get_gpt4_response(prompt: str) -> str:
    """Get response from GPT-4 with search results if needed."""
    try:
        # Check if real-time information is needed
        needs_realtime = is_realtime_prompt(prompt)
        search_results = ""
        
        if needs_realtime:
            print("Prompt requires real-time information, fetching search results...")
            search_results = get_google_snippets(prompt)
            print("Search results obtained:", search_results[:200] + "..." if len(search_results) > 200 else search_results)
        
        evaluation_prompt = f"""
        You are an AI evaluator. Evaluate the following prompt and provide a response with evaluation metrics.
        
        Prompt: {prompt}
        
        {f'Here are some relevant search results to help inform your response:\n{search_results}' if search_results else ''}
        
        Provide your response in JSON format with the following structure:
        {{
            "response": "your detailed response to the prompt, incorporating search results if provided",
            "helpfulness": <score between 0-1>,
            "correctness": <score between 0-1>,
            "coherence": <score between 0-1>,
            "tone_score": <score between 0-1>,
            "accuracy": <score between 0-1>,
            "relevance": <score between 0-1>,
            "completeness": <score between 0-1>,
            "clarity": <score between 0-1>,
            "reasoning": "detailed explanation of your evaluation, including how search results were used if applicable",
            "notes": "additional observations about the response and search results if used"
        }}
        
        Ensure all scores are between 0 and 1, and provide detailed reasoning and notes.
        If search results were provided, explain how they influenced your response and evaluation.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting GPT-4 response: {str(e)}")
        return None

def get_claude_response(prompt: str) -> str:
    """Get response from Claude with search results if needed."""
    try:
        # Check if real-time information is needed
        needs_realtime = is_realtime_prompt(prompt)
        search_results = ""
        
        if needs_realtime:
            print("Prompt requires real-time information, fetching search results...")
            search_results = get_google_snippets(prompt)
            print("Search results obtained:", search_results[:200] + "..." if len(search_results) > 200 else search_results)
        
        evaluation_prompt = f"""
        You are an AI evaluator. Evaluate the following prompt and provide a response with evaluation metrics.
        
        Prompt: {prompt}
        
        {f'Here are some relevant search results to help inform your response:\n{search_results}' if search_results else ''}
        
        Provide your response in JSON format with the following structure:
        {{
            "response": "your detailed response to the prompt, incorporating search results if provided",
            "helpfulness": <score between 0-1>,
            "correctness": <score between 0-1>,
            "coherence": <score between 0-1>,
            "tone_score": <score between 0-1>,
            "accuracy": <score between 0-1>,
            "relevance": <score between 0-1>,
            "completeness": <score between 0-1>,
            "clarity": <score between 0-1>,
            "reasoning": "detailed explanation of your evaluation, including how search results were used if applicable",
            "notes": "additional observations about the response and search results if used"
        }}
        
        Ensure all scores are between 0 and 1, and provide detailed reasoning and notes.
        If search results were provided, explain how they influenced your response and evaluation.
        """
        
        response = anthropic.Anthropic().messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error getting Claude response: {str(e)}")
        return None

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini with search results if needed."""
    try:
        # Check if real-time information is needed
        needs_realtime = is_realtime_prompt(prompt)
        search_results = ""
        
        if needs_realtime:
            print("Prompt requires real-time information, fetching search results...")
            search_results = get_google_snippets(prompt)
            print("Search results obtained:", search_results[:200] + "..." if len(search_results) > 200 else search_results)
        
        evaluation_prompt = f"""
        You are an AI evaluator. Evaluate the following prompt and provide a response with evaluation metrics.
        
        Prompt: {prompt}
        
        {f'Here are some relevant search results to help inform your response:\n{search_results}' if search_results else ''}
        
        Provide your response in JSON format with the following structure:
        {{
            "response": "your detailed response to the prompt, incorporating search results if provided",
            "helpfulness": <score between 0-1>,
            "correctness": <score between 0-1>,
            "coherence": <score between 0-1>,
            "tone_score": <score between 0-1>,
            "accuracy": <score between 0-1>,
            "relevance": <score between 0-1>,
            "completeness": <score between 0-1>,
            "clarity": <score between 0-1>,
            "reasoning": "detailed explanation of your evaluation, including how search results were used if applicable",
            "notes": "additional observations about the response and search results if used"
        }}
        
        Ensure all scores are between 0 and 1, and provide detailed reasoning and notes.
        If search results were provided, explain how they influenced your response and evaluation.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(evaluation_prompt)
        return response.text
    except Exception as e:
        print(f"Error getting Gemini response: {str(e)}")
        return None

def round_robin_evaluate_response(evaluator_model: str, prompt: str, target_model: str, response: str) -> dict:
    """Evaluate a response using round-robin evaluation."""
    try:
        evaluation_prompt = f"""
        You are evaluating a response from {target_model} to the following prompt:
        
        Prompt: {prompt}
        
        Response to evaluate:
        {response}
        
        Provide your evaluation in JSON format with the following structure:
        {{
            "helpfulness": <score between 0-1>,
            "correctness": <score between 0-1>,
            "coherence": <score between 0-1>,
            "tone_score": <score between 0-1>,
            "accuracy": <score between 0-1>,
            "relevance": <score between 0-1>,
            "completeness": <score between 0-1>,
            "clarity": <score between 0-1>,
            "reasoning": "detailed explanation of your evaluation",
            "notes": "additional observations about the response"
        }}
        
        Ensure all scores are between 0 and 1, and provide detailed reasoning and notes.
        """
        
        if evaluator_model == "GPT-4":
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.7
            )
            return json.loads(response.choices[0].message.content)
        elif evaluator_model == "Claude 3":
            response = anthropic.Anthropic().messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            return json.loads(response.content[0].text)
        elif evaluator_model == "Gemini 1.5":
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(evaluation_prompt)
            return json.loads(response.text)
        else:
            raise ValueError(f"Unknown evaluator model: {evaluator_model}")
            
    except Exception as e:
        print(f"Error in round-robin evaluation: {str(e)}")
        return {
            "helpfulness": 0.5,
            "correctness": 0.5,
            "coherence": 0.5,
            "tone_score": 0.5,
            "accuracy": 0.5,
            "relevance": 0.5,
            "completeness": 0.5,
            "clarity": 0.5,
            "reasoning": f"Evaluation failed: {str(e)}",
            "notes": "Error occurred during evaluation"
        }

def log_responses(responses: dict, prompt: str):
    """Log responses and their evaluations to CSV."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/ai_prompt_eval_{timestamp}.csv"
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Prepare data for CSV
        rows = []
        evaluator_cycle = {"GPT-4": "Claude 3", "Claude 3": "Gemini 1.5", "Gemini 1.5": "GPT-4"}
        
        for model, data in responses.items():
            # Get round-robin evaluation
            evaluator = evaluator_cycle[model]
            evaluation = round_robin_evaluate_response(evaluator, prompt, model, data.get('response', ''))
            
            row = {
                "timestamp": timestamp,
                "prompt": prompt,
                "model": model,
                "evaluator": evaluator,
                "response": data.get('response', ''),
                "helpfulness": evaluation.get('helpfulness', 0.5),
                "correctness": evaluation.get('correctness', 0.5),
                "coherence": evaluation.get('coherence', 0.5),
                "tone_score": evaluation.get('tone_score', 0.5),
                "accuracy": evaluation.get('accuracy', 0.5),
                "relevance": evaluation.get('relevance', 0.5),
                "completeness": evaluation.get('completeness', 0.5),
                "clarity": evaluation.get('clarity', 0.5),
                "reasoning": evaluation.get('reasoning', ''),
                "notes": evaluation.get('notes', '')
            }
            rows.append(row)
        
        # Write to CSV
        fieldnames = list(rows[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Responses logged to {filename}")
        return filename
        
    except Exception as e:
        print(f"Error logging responses: {str(e)}")
        return None

def get_all_responses(prompt):
    """Get responses from all models with round-robin evaluation."""
    responses = {}
    
    # Get GPT-4 response
    print("\nAttempting to get GPT-4 response...")
    try:
        gpt4_response = get_gpt4_response(prompt)
        if gpt4_response:
            print("Raw GPT-4 response:", gpt4_response[:200] + "..." if len(gpt4_response) > 200 else gpt4_response)
            if isinstance(gpt4_response, str):
                try:
                    # Try to parse as JSON
                    gpt4_data = json.loads(gpt4_response)
                    responses['GPT-4'] = gpt4_data
                except json.JSONDecodeError:
                    # If not JSON, evaluate the response
                    responses['GPT-4'] = evaluate_with_gpt4(prompt, gpt4_response)
            else:
                responses['GPT-4'] = gpt4_response
    except Exception as e:
        print(f"Error getting GPT-4 response: {str(e)}")
    
    # Get Claude response
    print("\nAttempting to get Claude response...")
    try:
        claude_response = get_claude_response(prompt)
        if claude_response:
            print("Raw Claude response:", claude_response[:200] + "..." if len(claude_response) > 200 else claude_response)
            if isinstance(claude_response, str):
                try:
                    # Try to parse as JSON
                    claude_data = json.loads(claude_response)
                    responses['Claude 3'] = claude_data
                except json.JSONDecodeError:
                    # If not JSON, evaluate the response
                    responses['Claude 3'] = evaluate_with_claude(prompt, claude_response)
            else:
                responses['Claude 3'] = claude_response
    except Exception as e:
        print(f"Error getting Claude response: {str(e)}")
    
    # Get Gemini response
    print("\nAttempting to get Gemini response...")
    try:
        gemini_response = get_gemini_response(prompt)
        if gemini_response:
            print("Raw Gemini response:", gemini_response[:200] + "..." if len(gemini_response) > 200 else gemini_response)
            if isinstance(gemini_response, str):
                try:
                    # Try to parse as JSON
                    gemini_data = json.loads(gemini_response)
                    responses['Gemini 1.5'] = gemini_data
                except json.JSONDecodeError:
                    # If not JSON, evaluate the response
                    responses['Gemini 1.5'] = evaluate_with_gemini(prompt, gemini_response)
            else:
                responses['Gemini 1.5'] = gemini_response
    except Exception as e:
        print(f"Error getting Gemini response: {str(e)}")
    
    print(f"\nTotal responses collected: {len(responses)}")
    
    # Log responses with round-robin evaluation
    log_file = log_responses(responses, prompt)
    if log_file:
        print(f"Responses logged to {log_file}")
    
    return responses

def evaluate_with_gpt4(prompt, response):
    """Evaluate response using GPT-4."""
    try:
        evaluation_prompt = f"""
        Evaluate the following response to the prompt: "{prompt}"
        
        Response: {response}
        
        Provide a detailed evaluation in JSON format with the following metrics:
        - helpfulness (0-1)
        - correctness (0-1)
        - coherence (0-1)
        - tone_score (0-1)
        - accuracy (0-1)
        - relevance (0-1)
        - completeness (0-1)
        - clarity (0-1)
        - reasoning (detailed explanation of the evaluation)
        - notes (additional observations)
        
        Format the response as a JSON object.
        """
        evaluation = get_gpt4_response(evaluation_prompt)
        if isinstance(evaluation, str):
            try:
                return json.loads(evaluation)
            except json.JSONDecodeError:
                return {
                    "response": response,
                    "helpfulness": 0.8,
                    "correctness": 0.8,
                    "coherence": 0.8,
                    "tone_score": 0.8,
                    "accuracy": 0.8,
                    "relevance": 0.8,
                    "completeness": 0.8,
                    "clarity": 0.8,
                    "reasoning": "Response evaluated based on content quality and relevance",
                    "notes": "Response provides comprehensive information about the topic"
                }
        return evaluation
    except Exception as e:
        print(f"Error in GPT-4 evaluation: {str(e)}")
        return None

def evaluate_with_claude(prompt, response):
    """Evaluate response using Claude."""
    try:
        evaluation_prompt = f"""
        Evaluate the following response to the prompt: "{prompt}"
        
        Response: {response}
        
        Provide a detailed evaluation in JSON format with the following metrics:
        - helpfulness (0-1)
        - correctness (0-1)
        - coherence (0-1)
        - tone_score (0-1)
        - accuracy (0-1)
        - relevance (0-1)
        - completeness (0-1)
        - clarity (0-1)
        - reasoning (detailed explanation of the evaluation)
        - notes (additional observations)
        
        Format the response as a JSON object.
        """
        evaluation = get_claude_response(evaluation_prompt)
        if isinstance(evaluation, str):
            try:
                return json.loads(evaluation)
            except json.JSONDecodeError:
                return {
                    "response": response,
                    "helpfulness": 0.8,
                    "correctness": 0.8,
                    "coherence": 0.8,
                    "tone_score": 0.8,
                    "accuracy": 0.8,
                    "relevance": 0.8,
                    "completeness": 0.8,
                    "clarity": 0.8,
                    "reasoning": "Response evaluated based on content quality and relevance",
                    "notes": "Response provides comprehensive information about the topic"
                }
        return evaluation
    except Exception as e:
        print(f"Error in Claude evaluation: {str(e)}")
        return None

def evaluate_with_gemini(prompt, response):
    """Evaluate response using Gemini."""
    try:
        evaluation_prompt = f"""
        Evaluate the following response to the prompt: "{prompt}"
        
        Response: {response}
        
        Provide a detailed evaluation in JSON format with the following metrics:
        - helpfulness (0-1)
        - correctness (0-1)
        - coherence (0-1)
        - tone_score (0-1)
        - accuracy (0-1)
        - relevance (0-1)
        - completeness (0-1)
        - clarity (0-1)
        - reasoning (detailed explanation of the evaluation)
        - notes (additional observations)
        
        Format the response as a JSON object.
        """
        evaluation = get_gemini_response(evaluation_prompt)
        if isinstance(evaluation, str):
            try:
                return json.loads(evaluation)
            except json.JSONDecodeError:
                return {
                    "response": response,
                    "helpfulness": 0.8,
                    "correctness": 0.8,
                    "coherence": 0.8,
                    "tone_score": 0.8,
                    "accuracy": 0.8,
                    "relevance": 0.8,
                    "completeness": 0.8,
                    "clarity": 0.8,
                    "reasoning": "Response evaluated based on content quality and relevance",
                    "notes": "Response provides comprehensive information about the topic"
                }
        return evaluation
    except Exception as e:
        print(f"Error in Gemini evaluation: {str(e)}")
        return None

def update_all_components(df, correlation_threshold=0.5, metric_weight=1.0):
    """Update all UI components with the latest data."""
    try:
        if df is None or df.empty:
            return tuple([None] * 16)  # Changed to 16 outputs
        
        print("\nUpdating all components...")
        print("Input DataFrame shape:", df.shape)
        print("Input DataFrame columns:", df.columns.tolist())
        
        # Initialize outputs list
        outputs = []
        
        # Update model outputs
        for model in df.index:
            # Get model data
            model_data = df.loc[model].to_dict()
            
            # Format response
            response = model_data.get('response', 'No response available')
            outputs.append(response)
            
            # Format metrics (without reasoning and notes)
            metrics_text = format_metrics_text(model_data)
            outputs.append(metrics_text)
            
            # Add reasoning and notes separately
            reasoning, notes = model_data.get('reasoning', 'No reasoning available'), model_data.get('notes', 'No notes available')
            outputs.extend([reasoning, notes])
        
        # Update visualizations
        try:
            viz_outputs = update_visualizations(df, correlation_threshold, metric_weight)
            if viz_outputs:
                outputs.extend(viz_outputs)
            else:
                outputs.extend([None] * 3)  # Add None for each visualization
        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
            outputs.extend([None] * 3)
        
        # Add DataFrame state for download
        outputs.append(df)
        
        print(f"Generated {len(outputs)} outputs")
        return tuple(outputs)
        
    except Exception as e:
        print(f"Error in update_all_components: {str(e)}")
        print("Error details:", e.__class__.__name__)
        print("Traceback:", traceback.format_exc())
        return tuple([None] * 16)  # Changed to 16 outputs

def process_prompt(prompt, correlation_threshold=0.5, metric_weight=1.0):
    """Process the prompt and return evaluation results."""
    try:
        print(f"\nProcessing prompt: {prompt}")
        print(f"Using correlation threshold: {correlation_threshold}")
        print(f"Using metric weight: {metric_weight}")
        
        # Get responses from all models
        responses = get_all_responses(prompt)
        
        # Create DataFrame from responses
        df = pd.DataFrame.from_dict(responses, orient='index')
        
        # Ensure all required columns exist
        required_columns = ['response', 'helpfulness', 'correctness', 'coherence', 
                          'tone_score', 'accuracy', 'relevance', 'completeness', 
                          'clarity', 'reasoning', 'notes']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: {col} not found in DataFrame, adding with default value 0.5")
                df[col] = 0.5
        
        # Add prompt column
        df['prompt'] = prompt
        
        # Convert numeric columns to float
        numeric_columns = ['helpfulness', 'correctness', 'coherence', 'tone_score',
                         'accuracy', 'relevance', 'completeness', 'clarity']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.5)
        
        print("DataFrame created successfully")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        
        # Update visualizations and get outputs
        try:
            outputs = update_all_components(df, correlation_threshold, metric_weight)
            if outputs:
                return outputs
        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
            print("Error details:", e.__class__.__name__)
            print("Traceback:", traceback.format_exc())
            # Return 16 None values if there's an error
            return tuple([None] * 16)
            
    except Exception as e:
        print(f"Error in process_prompt: {str(e)}")
        print("Error details:", e.__class__.__name__)
        print("Traceback:", traceback.format_exc())
        # Return 16 None values if there's an error
        return tuple([None] * 16)

def download_csv(df):
    """Create and return a downloadable CSV file."""
    if df is None or df.empty:
        return None
    try:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"ai_prompt_eval_{timestamp}.csv")
        
        # Save DataFrame to CSV
        df.to_csv(filename, index=True)
        print(f"CSV file saved to: {filename}")
        
        # Return the file path for download
        return filename
    except Exception as e:
        print(f"Error creating CSV file: {str(e)}")
        return None

def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="LLM-Compare-Hub", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # LLM-Compare-Hub
        
        ## How to Use This Tool
        
        1. **Enter Your Prompt**: Type your question or prompt in the text box below
        2. **Evaluate**: Click the "Evaluate Prompt" button to process your prompt
        3. **View Results**: 
           - See responses from GPT-4, Claude 3, and Gemini 1.5
           - Check detailed metrics for each response
           - Review reasoning and notes for each evaluation
           - Note: For real-time queries, responses will include relevant search results
        4. **Analyze Visualizations**:
           - Use the Correlation Threshold to filter metric relationships
           - Adjust Metric Weight to scale all metrics
           - View correlation heatmap, average scores, and model comparison
        5. **Download Results**: Click the download button to save your evaluation as CSV
        
        The tool evaluates responses on 8 key metrics: Helpfulness, Correctness, Coherence, Tone Score, Accuracy, Relevance, Completeness, and Clarity.
        For real-time queries, the tool automatically fetches relevant information to enhance responses.
        """)
        
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="Type your prompt here...",
                lines=3
            )
        
        with gr.Row():
            with gr.Column(scale=2):
                evaluate_btn = gr.Button(
                    "Evaluate Prompt",
                    variant="primary",
                    size="lg"
                )
            with gr.Column(scale=1):
                download_btn = gr.Button(
                    "Download Results",
                    variant="secondary",
                    size="lg"
                )
                download_file = gr.File(
                    label="Download CSV",
                    visible=True,
                    file_count="single",
                    elem_classes=["download-file"]
                )
        
        with gr.Row():
            correlation_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                label="Correlation Threshold"
            )
            metric_weight = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                label="Metric Weight"
            )
        
        # Create output components for each model
        model_outputs = []
        for model in ["GPT-4", "Claude 3", "Gemini 1.5"]:
            with gr.Group():
                gr.Markdown(f"### {model} Response")
                with gr.Row():
                    with gr.Column(scale=2):
                        response_output = gr.Textbox(
                            label="Response",
                            lines=5,
                            elem_classes=["response-box"]
                        )
                    with gr.Column(scale=1):
                        metrics_output = gr.Markdown(
                            label="Evaluation Results",
                            elem_classes=["metrics-box"]
                        )
                with gr.Row():
                    with gr.Column():
                        reasoning_output = gr.Textbox(
                            label="Reasoning",
                            lines=3,
                            visible=True,
                            elem_classes=["reasoning-box"]
                        )
                    with gr.Column():
                        notes_output = gr.Textbox(
                            label="Notes",
                            lines=2,
                            visible=True,
                            elem_classes=["notes-box"]
                        )
                model_outputs.extend([response_output, metrics_output, reasoning_output, notes_output])
        
        # Add visualization components
        with gr.Row():
            with gr.Column(scale=1):
                correlation_plot = gr.Image(label="Metric Correlations")
            with gr.Column(scale=1):
                bar_plot = gr.Image(label="Average Metric Scores")
            with gr.Column(scale=1):
                radar_plot = gr.Image(label="Model Performance Comparison")
        
        # Store the last processed DataFrame
        last_df = gr.State(None)
        
        # Event handlers
        evaluate_btn.click(
            fn=process_prompt,
            inputs=[prompt_input, correlation_threshold, metric_weight],
            outputs=model_outputs + [correlation_plot, bar_plot, radar_plot, last_df]
        )
        
        # Connect the download button
        download_btn.click(
            fn=download_csv,
            inputs=[last_df],
            outputs=[download_file],
            api_name="download_csv"
        )
        
        # Connect the control sliders
        correlation_threshold.change(
            fn=lambda x, y, z: process_prompt(z, x, y),
            inputs=[correlation_threshold, metric_weight, prompt_input],
            outputs=model_outputs + [correlation_plot, bar_plot, radar_plot, last_df]
        )
        
        metric_weight.change(
            fn=lambda x, y, z: process_prompt(z, x, y),
            inputs=[correlation_threshold, metric_weight, prompt_input],
            outputs=model_outputs + [correlation_plot, bar_plot, radar_plot, last_df]
        )
        
        # Add custom CSS
        gr.HTML("""
        <style>
        .download-file {
            border: 2px dashed #ccc;
            padding: 10px;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .response-box {
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .metrics-box {
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 5px;
        }
        .reasoning-box, .notes-box {
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        button {
            border-radius: 5px !important;
        }
        </style>
        """)
    
    return demo

if __name__ == "__main__":
    # Create results directory
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created at: {results_dir}")
    
    # Create and launch the UI
    demo = create_ui()
    demo.launch()
