import os
from openai import OpenAI
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import csv
import json

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def evaluate_response(evaluator_model, prompt, target_model, response_text):
    """Evaluate a response using the specified evaluator model."""
    evaluation_prompt = (
        f"You are an AI tasked with evaluating another model's response.\n"
        f"Here is the original prompt: \"{prompt}\"\n"
        f"Here is the response from {target_model}: \"{response_text}\"\n\n"
        f"Evaluate this response on the following criteria from 0 (worst) to 1 (best):\n"
        f"- Helpfulness: How useful and informative is the response?\n"
        f"- Correctness: How accurate and factually correct is the response?\n"
        f"- Coherence: How well-structured and logical is the response?\n"
        f"- Tone: How appropriate and professional is the tone?\n"
        f"- Accuracy: How precise and detailed is the information?\n"
        f"- Relevance: How well does the response address the prompt?\n"
        f"- Completeness: How comprehensive is the response?\n"
        f"- Clarity: How clear and easy to understand is the response?\n\n"
        f"Return the result in this exact JSON format:\n\n"
        f"{{\n"
        f"  \"helpfulness\": <0-1>,\n"
        f"  \"correctness\": <0-1>,\n"
        f"  \"coherence\": <0-1>,\n"
        f"  \"tone_score\": <0-1>,\n"
        f"  \"accuracy\": <0-1>,\n"
        f"  \"relevance\": <0-1>,\n"
        f"  \"completeness\": <0-1>,\n"
        f"  \"clarity\": <0-1>,\n"
        f"  \"reasoning\": \"detailed explanation for the scores\",\n"
        f"  \"notes\": \"additional observations about the response\"\n"
        f"}}"
    )

    try:
        if evaluator_model == "GPT-4":
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.3
            )
            result = response.choices[0].message.content
        elif evaluator_model == "Claude 3":
            response = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            result = response.content[0].text
        elif evaluator_model == "Gemini 1.5":
            model = genai.GenerativeModel("gemini-1.5-pro")
            eval_response = model.generate_content(evaluation_prompt)
            result = eval_response.text
        else:
            print(f"Unknown evaluator model: {evaluator_model}")
            return None
        
        # Try to parse JSON response
        try:
            if isinstance(result, str):
                parsed = json.loads(result)
            else:
                parsed = result
            return parsed
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from {evaluator_model} evaluation")
            return None
            
    except Exception as e:
        print(f"Error in {evaluator_model} evaluation: {str(e)}")
        return None

def comprehensive_round_robin_evaluation(responses_dict, prompt):
    """
    Perform comprehensive round-robin evaluation where each model evaluates all other models.
    
    Args:
        responses_dict: Dictionary with model names as keys and response texts as values
        prompt: The original prompt
    
    Returns:
        Dictionary with comprehensive evaluation results
    """
    print("\nStarting comprehensive round-robin evaluation...")
    
    # Define the evaluation matrix
    evaluation_matrix = {
        "GPT-4": ["Claude 3", "Gemini 1.5"],
        "Claude 3": ["GPT-4", "Gemini 1.5"], 
        "Gemini 1.5": ["GPT-4", "Claude 3"]
    }
    
    # Initialize results structure
    comprehensive_results = {}
    
    # For each model, collect evaluations from other models
    for target_model, response_text in responses_dict.items():
        print(f"\nCollecting evaluations for {target_model}...")
        
        # Initialize target model data
        comprehensive_results[target_model] = {
            'response': response_text,
            'evaluations': {},
            'average_scores': {}
        }
        
        # Get evaluations from other models
        evaluators = evaluation_matrix[target_model]
        for evaluator in evaluators:
            print(f"  {evaluator} evaluating {target_model}...")
            evaluation = evaluate_response(evaluator, prompt, target_model, response_text)
            
            if evaluation:
                comprehensive_results[target_model]['evaluations'][evaluator] = evaluation
                print(f"    {evaluator} evaluation completed")
            else:
                print(f"    {evaluator} evaluation failed")
        
        # Calculate average scores across all evaluators
        if comprehensive_results[target_model]['evaluations']:
            metrics = ['helpfulness', 'correctness', 'coherence', 'tone_score', 
                      'accuracy', 'relevance', 'completeness', 'clarity']
            
            for metric in metrics:
                scores = []
                for evaluator, eval_data in comprehensive_results[target_model]['evaluations'].items():
                    if metric in eval_data and isinstance(eval_data[metric], (int, float)):
                        scores.append(eval_data[metric])
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    comprehensive_results[target_model]['average_scores'][metric] = round(avg_score, 3)
                else:
                    comprehensive_results[target_model]['average_scores'][metric] = 0.5
    
    print(f"\nComprehensive evaluation completed for {len(comprehensive_results)} models")
    return comprehensive_results

def save_comprehensive_results(comprehensive_results, prompt, timestamp=None):
    """Save comprehensive evaluation results to CSV."""
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"results/comprehensive_eval_{timestamp}.csv"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Prepare data for CSV
    rows = []
    for model, data in comprehensive_results.items():
        # Get average scores
        avg_scores = data.get('average_scores', {})
        
        # Create row for each evaluator
        for evaluator, evaluation in data.get('evaluations', {}).items():
            row = {
                'timestamp': timestamp,
                'prompt': prompt,
                'target_model': model,
                'evaluator': evaluator,
                'response': data['response'],
                'helpfulness': evaluation.get('helpfulness', 0.5),
                'correctness': evaluation.get('correctness', 0.5),
                'coherence': evaluation.get('coherence', 0.5),
                'tone_score': evaluation.get('tone_score', 0.5),
                'accuracy': evaluation.get('accuracy', 0.5),
                'relevance': evaluation.get('relevance', 0.5),
                'completeness': evaluation.get('completeness', 0.5),
                'clarity': evaluation.get('clarity', 0.5),
                'reasoning': evaluation.get('reasoning', ''),
                'notes': evaluation.get('notes', ''),
                'avg_helpfulness': avg_scores.get('helpfulness', 0.5),
                'avg_correctness': avg_scores.get('correctness', 0.5),
                'avg_coherence': avg_scores.get('coherence', 0.5),
                'avg_tone_score': avg_scores.get('tone_score', 0.5),
                'avg_accuracy': avg_scores.get('accuracy', 0.5),
                'avg_relevance': avg_scores.get('relevance', 0.5),
                'avg_completeness': avg_scores.get('completeness', 0.5),
                'avg_clarity': avg_scores.get('clarity', 0.5)
            }
            rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Results saved to {filename}")
        return filename
    else:
        print("No results to save")
        return None

def round_robin_evaluate_and_log(responses):
    """Legacy function for backward compatibility."""
    print("This function is deprecated. Use comprehensive_round_robin_evaluation instead.")
    return comprehensive_round_robin_evaluation(responses, "Legacy prompt")

if __name__ == "__main__":
    # Test the evaluation system
    test_responses = {
        "GPT-4": "This is a test response from GPT-4.",
        "Claude 3": "This is a test response from Claude 3.",
        "Gemini 1.5": "This is a test response from Gemini 1.5."
    }
    
    test_prompt = "What is artificial intelligence?"
    
    print("Testing round-robin evaluation system...")
    results = comprehensive_round_robin_evaluation(test_responses, test_prompt)
    
    if results:
        print("\nTest completed successfully!")
        for model, data in results.items():
            print(f"\n{model} average scores:")
            for metric, score in data.get('average_scores', {}).items():
                print(f"  {metric}: {score}")
    else:
        print("Test failed!")
