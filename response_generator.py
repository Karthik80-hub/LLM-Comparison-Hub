import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai

# Load API keys from .env
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gpt4_response(prompt):
    try:
        if "Recent info:" in prompt:
            user_prompt, realtime_info = prompt.split("Recent info:", 1)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert ATS evaluator. You are comparing a job description (JD) and a resume to produce an ATS score. "
                        "Highlight matches, gaps, suggestions for improvement, and an overall score."
                    )
                },
                {"role": "user", "content": user_prompt.strip()},
                {
                    "role": "user",
                    "content": (
                        f"Here is some recent real-time context for your reference:\n\n{realtime_info.strip()}\n\n"
                        "Based on this, tailor your response as if the data is accurate."
                    )
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert ATS evaluator. You are comparing a job description (JD) and a resume to produce an ATS score. "
                        "Highlight matches, gaps, suggestions for improvement, and an overall score."
                    )
                },
                {"role": "user", "content": prompt}
            ]

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error with GPT-4: {e}")
        return "GPT-4 failed."

def get_claude_response(prompt):
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
        return "Claude 3 failed."

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error with Gemini: {e}")
        return "Gemini 1.5 failed."

def generate_all_responses_with_reasoning(prompt, selected_models=None):
    all_models = {
        "GPT-4": get_gpt4_response,
        "Claude 3": get_claude_response,
        "Gemini 1.5": get_gemini_response
    }
    models_to_use = selected_models if selected_models else list(all_models.keys())

    responses = {}
    for model_name in models_to_use:
        fetch_fn = all_models[model_name]
        try:
            response = fetch_fn(prompt)
            reason_prompt = (
                f"Why did you generate this response to the prompt:\n\n"
                f"\"{prompt}\"\n\n"
                f"Your Response:\n\"{response}\"\n\n"
                "Explain your reasoning behind structuring or phrasing it that way."
            )
            reasoning = fetch_fn(reason_prompt)
            responses[model_name] = {"response": response, "reasoning": reasoning}
        except Exception as e:
            responses[model_name] = {"response": "Failed", "reasoning": str(e)}

    return responses
