import csv
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
from round_robin_evaluator import comprehensive_round_robin_evaluation, save_comprehensive_results
from datetime import datetime

# Load API keys from .env
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gpt4_response(prompt):
    """Get response from GPT-4."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with GPT-4: {e}")
        return None

def get_claude_response(prompt):
    """Get response from Claude."""
    try:
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error with Claude 3: {e}")
        return None

def get_gemini_response(prompt):
    """Get response from Gemini."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error with Gemini: {e}")
        return None

def display_responses_side_by_side(responses, prompt):
    """Display responses in a formatted side-by-side comparison."""
    print("\n" + "="*80)
    print(f"PROMPT: {prompt}")
    print("="*80)
    
    models = list(responses.keys())
    if len(models) == 0:
        print("No responses to display")
        return
    
    # Display responses
    for i, model in enumerate(models, 1):
        response = responses[model]
        print(f"\n{i}. {model} RESPONSE:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print(f"Length: {len(response)} characters")
        print()

def generate_and_compare_responses():
    """Generate responses from all models and display comparison."""
    print("=== Response Generator - Model Comparison Tool ===\n")
    
    # Get prompt from user
    prompt = input("Enter your prompt: ")
    if not prompt.strip():
        print("No prompt provided. Exiting.")
        return
    
    print(f"\nGenerating responses for: '{prompt}'")
    print("=" * 60)
    
    # Collect responses from all models
    responses = {}
    
    print("\n1. Generating GPT-4 response...")
    gpt_response = get_gpt4_response(prompt)
    if gpt_response:
        responses['GPT-4'] = gpt_response
        print("GPT-4 response generated")
    else:
        print("GPT-4 failed")
    
    print("\n2. Generating Claude response...")
    claude_response = get_claude_response(prompt)
    if claude_response:
        responses['Claude 3'] = claude_response
        print("Claude response generated")
    else:
        print("Claude failed")
    
    print("\n3. Generating Gemini response...")
    gemini_response = get_gemini_response(prompt)
    if gemini_response:
        responses['Gemini 1.5'] = gemini_response
        print("Gemini response generated")
    else:
        print("Gemini failed")
    
    if not responses:
        print("\nNo models generated responses. Check your API keys.")
        return
    
    print(f"\nSuccessfully generated {len(responses)} responses")
    
    # Display side-by-side comparison
    display_responses_side_by_side(responses, prompt)
    
    # Ask if user wants evaluation
    evaluate = input("\nDo you want to evaluate these responses? (y/n): ").strip().lower()
    
    if evaluate in ['y', 'yes']:
        print("\n4. Performing comprehensive evaluation...")
        try:
            comprehensive_results = comprehensive_round_robin_evaluation(responses, prompt)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = save_comprehensive_results(comprehensive_results, prompt, timestamp)
            
            if csv_file:
                print(f"Evaluation results saved to: {csv_file}")
            
            # Display evaluation summary
            print("\n=== EVALUATION SUMMARY ===")
            for model, data in comprehensive_results.items():
                avg_scores = data.get('average_scores', {})
                print(f"\n{model} Scores:")
                print(f"  Helpfulness: {avg_scores.get('helpfulness', 'N/A')}")
                print(f"  Correctness: {avg_scores.get('correctness', 'N/A')}")
                print(f"  Coherence: {avg_scores.get('coherence', 'N/A')}")
                print(f"  Clarity: {avg_scores.get('clarity', 'N/A')}")
                print(f"  Evaluated by: {list(data.get('evaluations', {}).keys())}")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
    
    print("\n=== Response generation completed ===")

def batch_generate_from_file(filename):
    """Generate responses for multiple prompts from a file."""
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return
    
    print(f"=== Batch Response Generation from {filename} ===")
    
    with open(filename, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(prompts)} prompts to process")
    
    all_results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Processing Prompt {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        
        # Generate responses
        responses = {}
        
        gpt_response = get_gpt4_response(prompt)
        if gpt_response:
            responses['GPT-4'] = gpt_response
        
        claude_response = get_claude_response(prompt)
        if claude_response:
            responses['Claude 3'] = claude_response
        
        gemini_response = get_gemini_response(prompt)
        if gemini_response:
            responses['Gemini 1.5'] = gemini_response
        
        if responses:
            # Display comparison
            display_responses_side_by_side(responses, prompt)
            
            # Evaluate
            try:
                comprehensive_results = comprehensive_round_robin_evaluation(responses, prompt)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = save_comprehensive_results(comprehensive_results, prompt, f"{timestamp}_batch_{i}")
                print(f"Results saved for prompt {i}")
                all_results.append((prompt, comprehensive_results))
            except Exception as e:
                print(f"Evaluation failed for prompt {i}: {e}")
        else:
            print(f"No responses for prompt {i}")
    
    # Save summary
    if all_results:
        summary_file = f"results/batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("results", exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("BATCH RESPONSE GENERATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for prompt, results in all_results:
                f.write(f"PROMPT: {prompt}\n")
                f.write("-" * 30 + "\n")
                
                for model, data in results.items():
                    avg_scores = data.get('average_scores', {})
                    f.write(f"{model}:\n")
                    f.write(f"  Helpfulness: {avg_scores.get('helpfulness', 'N/A')}\n")
                    f.write(f"  Correctness: {avg_scores.get('correctness', 'N/A')}\n")
                    f.write(f"  Coherence: {avg_scores.get('coherence', 'N/A')}\n")
                    f.write(f"  Clarity: {avg_scores.get('clarity', 'N/A')}\n\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"\nBatch summary saved to: {summary_file}")
    
    print("\n=== Batch generation completed ===")

def generate_all_responses(prompt):
    """Generate responses from all models for a given prompt."""
    responses = {}
    
    # Generate responses from all models
    gpt_response = get_gpt4_response(prompt)
    if gpt_response:
        responses['GPT-4'] = gpt_response
    
    claude_response = get_claude_response(prompt)
    if claude_response:
        responses['Claude 3'] = claude_response
    
    gemini_response = get_gemini_response(prompt)
    if gemini_response:
        responses['Gemini 1.5'] = gemini_response
    
    return responses

if __name__ == "__main__":
    print("=== Response Generator Tool ===")
    print("1. Interactive mode")
    print("2. Batch mode from file")
    
    choice = input("Choose mode (1 or 2): ").strip()
    
    if choice == "1":
        generate_and_compare_responses()
    elif choice == "2":
        filename = input("Enter filename with prompts (one per line): ").strip()
        batch_generate_from_file(filename)
    else:
        print("Invalid choice. Exiting.")
