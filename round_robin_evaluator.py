
import os
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import csv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Round robin evaluator logic
def evaluate_response(evaluator_model, prompt, model_name, response_text):
    evaluation_prompt = (
        f"You are an AI tasked with evaluating another model's response.\n"
        f"Here is the original prompt: \"{prompt}\"\n"
        f"Here is the response from {model_name}: \"{response_text}\"\n\n"
        f"Evaluate this response on the following criteria from 1 (worst) to 5 (best):\n"
        f"- Helpfulness\n"
        f"- Correctness\n"
        f"- Coherence\n"
        f"- Tone\n\n"
        f"Also briefly explain each score. Return the result in this exact JSON format:\n\n"
        f"{{\n"
        f"  \"helpfulness\": <1-5>,\n"
        f"  \"correctness\": <1-5>,\n"
        f"  \"coherence\": <1-5>,\n"
        f"  \"tone_score\": <1-5>,\n"
        f"  \"reasoning\": \"brief explanation for the scores\"\n"
        f"}}"
    )

    if evaluator_model == "GPT-4":
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    elif evaluator_model == "Claude 3":
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        return response.content[0].text
    elif evaluator_model == "Gemini 1.5":
        model = genai.GenerativeModel("gemini-1.5-pro")
        eval_response = model.generate_content(evaluation_prompt)
        return eval_response.text

def round_robin_evaluate_and_log(responses):
    evaluator_cycle = {"GPT-4": "Claude 3", "Claude 3": "Gemini 1.5", "Gemini 1.5": "GPT-4"}
    evaluated_rows = []

    for r in responses:
        evaluator = evaluator_cycle[r["model"]]
        try:
            result = evaluate_response(evaluator, r["prompt"], r["model"], r["response"])
            parsed = eval(result) if isinstance(result, str) else result
            row = {
                "prompt": r["prompt"],
                "model": r["model"],
                "response": r["response"],
                "helpfulness": parsed.get("helpfulness"),
                "correctness": parsed.get("correctness"),
                "coherence": parsed.get("coherence"),
                "tone_score": parsed.get("tone_score"),
                "bias_flag": "",
                "notes": parsed.get("reasoning", "")
            }
            evaluated_rows.append(row)
        except Exception as e:
            print(f"Evaluation failed for {r['model']} by {evaluator}:", e)

    with open("ai_prompt_eval_template.csv", "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["prompt", "model", "response", "helpfulness", "correctness", "coherence", "tone_score", "bias_flag", "notes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for row in evaluated_rows:
            writer.writerow(row)

    print("All responses evaluated and saved with scores and reasoning.")
