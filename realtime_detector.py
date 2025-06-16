# realtime_detector.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_realtime_prompt(prompt: str) -> bool:
    try:
        system_msg = "You are a classifier that determines whether a user's question requires real-time information or not. Answer 'yes' or 'no'."
        user_msg = f"Question: {prompt}\nAnswer with yes or no:"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0
        )

        reply = response.choices[0].message.content.strip().lower()
        return "yes" in reply

    except Exception as e:
        print("[RealTime Detector Error]", e)
        return False
