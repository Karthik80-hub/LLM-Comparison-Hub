import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
from search_fallback import get_google_snippets

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def detect_realtime_prompt(prompt: str) -> bool:
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Respond only with 'True' or 'False'. Does the following prompt require real-time or up-to-date information to answer accurately?"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return "true" in resp.choices[0].message.content.strip().lower()
    except Exception as e:
        print("Realtime detection failed:", e)
        return False

def build_prompt(prompt, resume=None, jd=None, search_snippets=None):
    if resume and jd:
        return f"""### Task: Evaluate ATS Fit

Resume:
{resume}

Job Description:
{jd}

Instructions:
- Provide an ATS match score out of 100
- Justify the score with 3 bullet points
- Highlight missing skills, if any
"""
    if search_snippets:
        return f"{prompt}\n\n[Latest Info Retrieved via Google Search]\n{search_snippets}"
    return prompt

def ask_reasoning(model_name, response, prompt):
    follow_up = f"""You answered: 

{response}

Now explain why you gave this answer to the prompt: "{prompt}".
Respond in 2-3 bullet points."""
    try:
        if model_name == "GPT-4":
            resp = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're an AI that explains why a given answer was provided."},
                    {"role": "user", "content": follow_up}
                ],
                temperature=0
            )
            return resp.choices[0].message.content.strip()
        elif model_name == "Claude 3":
            resp = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": follow_up}]
            )
            return resp.content[0].text.strip()
        elif model_name == "Gemini 1.5":
            model = genai.GenerativeModel("gemini-1.5-pro")
            return model.generate_content(follow_up).text.strip()
    except Exception as e:
        return f"[Reasoning Error] {e}"

def get_gpt4_response(prompt, resume=None, jd=None, search_snippets=None):
    system_instruction = (
        "You are ChatGPT. If search results are included in the prompt, use them explicitly. "
        "Do not ignore them or hallucinate."
    )
    full_prompt = build_prompt(prompt, resume, jd, search_snippets)
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT-4 Error] {e}"

def get_claude_response(prompt, resume=None, jd=None, search_snippets=None):
    full_prompt = build_prompt(prompt, resume, jd, search_snippets)
    try:
        resp = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"[Claude Error] {e}"

def get_gemini_response(prompt, resume=None, jd=None, search_snippets=None):
    full_prompt = build_prompt(prompt, resume, jd, search_snippets)
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        return model.generate_content(full_prompt).text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

def universal_model_responses(prompt, resume=None, jd=None, selected_models=None):
    if selected_models is None:
        selected_models = ["GPT-4", "Claude 3", "Gemini 1.5"]

    is_ats = bool(resume and jd)

    # Use realtime search only if not ATS prompt
    search_snippets = None
    if not is_ats and detect_realtime_prompt(prompt):
        try:
            search_snippets = get_google_snippets(prompt)
        except Exception as e:
            print("[Search Fallback Error]", e)

    results = {}

    if "GPT-4" in selected_models:
        gpt_resp = get_gpt4_response(prompt, resume, jd, search_snippets)
        results["GPT-4"] = {
            "response": gpt_resp,
            "reasoning": ask_reasoning("GPT-4", gpt_resp, prompt),
            "search_results": search_snippets,
            "is_ats": is_ats
        }

    if "Claude 3" in selected_models:
        claude_resp = get_claude_response(prompt, resume, jd, search_snippets)
        results["Claude 3"] = {
            "response": claude_resp,
            "reasoning": ask_reasoning("Claude 3", claude_resp, prompt),
            "search_results": search_snippets,
            "is_ats": is_ats
        }

    if "Gemini 1.5" in selected_models:
        gemini_resp = get_gemini_response(prompt, resume, jd, search_snippets)
        results["Gemini 1.5"] = {
            "response": gemini_resp,
            "reasoning": ask_reasoning("Gemini 1.5", gemini_resp, prompt),
            "search_results": search_snippets,
            "is_ats": is_ats
        }

    return results
