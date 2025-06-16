
import csv
import os
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai

# Load API keys from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Prompt input
prompt = input("Enter your prompt: ")

# Initialize output list
responses = []

# OpenAI GPT-4
try:
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    responses.append({
        "prompt": prompt,
        "model": "GPT-4",
        "response": gpt_response['choices'][0]['message']['content'],
        "helpfulness": "",
        "correctness": "",
        "coherence": "",
        "tone_score": "",
        "bias_flag": "",
        "notes": ""
    })
except Exception as e:
    print("Error with GPT-4:", e)

# Claude 3
try:
    claude_response = anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    responses.append({
        "prompt": prompt,
        "model": "Claude 3",
        "response": claude_response.content[0].text,
        "helpfulness": "",
        "correctness": "",
        "coherence": "",
        "tone_score": "",
        "bias_flag": "",
        "notes": ""
    })
except Exception as e:
    print("Error with Claude 3:", e)

# Gemini 1.5
try:
    model = genai.GenerativeModel("gemini-1.5-pro")
    gemini_response = model.generate_content(prompt)
    responses.append({
        "prompt": prompt,
        "model": "Gemini 1.5",
        "response": gemini_response.text,
        "helpfulness": "",
        "correctness": "",
        "coherence": "",
        "tone_score": "",
        "bias_flag": "",
        "notes": ""
    })
except Exception as e:
    print("Error with Gemini:", e)

# Append results to CSV
with open("ai_prompt_eval_template.csv", "a", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["prompt", "model", "response", "helpfulness", "correctness", "coherence", "tone_score", "bias_flag", "notes"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for row in responses:
        writer.writerow(row)

print("\nResponses saved! Open ai_prompt_eval_template.csv to begin scoring.")
