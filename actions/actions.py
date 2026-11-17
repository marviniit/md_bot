import json
import re
import csv
import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# ==============================================================================
# LLM BACKEND CONFIGURATION (SWAPPABLE: ONLINE NVIDIA API vs LOCAL LM STUDIO)
# ==============================================================================
# Choose "online" (Preferred: NVIDIA Free API) OR "local" (Fallback: LM Studio Bionic)
LLM_MODE = "online"   # <-- Change to "local" to use LM Studio Bionic locally

# --- Option A: Online NVIDIA Free API (Preferred) ---
# 1. Get your free API key at: https://build.nvidia.com/settings/api-keys
# 2. Paste your key BELOW (do NOT use Windows Environment Variables for this lab).
# 3. NOTE: Free API models may occasionally experience downtime, rate limits, or updates.
#    If a model is down or returns an error, check https://build.nvidia.com/explore/discover
#    for available "Free Endpoint" models (e.g., "google/gemma-3-4b-it" or "google/gemma-4-31b-it")
#    and replace ONLINE_MODEL_NAME below, OR switch LLM_MODE = "local".
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = "nvapi-REPLACE_WITH_YOUR_API_KEY"  # <-- Paste your nvapi-... key here
ONLINE_MODEL_NAME = "z-ai/glm-5.3-flash"  # Alternatives: "google/gemma-3-4b-it", "google/gemma-4-31b-it"

# --- Option B: Local LM Studio Bionic API Server (Offline / Fallback) ---
# Ensure LM Studio Bionic Local Server is running on port 1234 with qwen/qwen3.5-2b loaded
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
LOCAL_MODEL_NAME = "qwen/qwen3.5-2b"      # Same compact Traditional Chinese model from WS08

# How long (seconds) to wait for the LLM API response
LLM_TIMEOUT = 180
# ==============================================================================


def _as_utf8_text(raw_line) -> Text:
    """Decode one SSE line as UTF-8. Do not let requests guess Latin-1."""
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8", errors="replace")
    return raw_line


def _repair_utf8_mojibake(text: Text) -> Text:
    """Turn 'æ¨å¥½' back into '您好' when UTF-8 was decoded as Latin-1/CP1252."""
    for encoding in ("latin-1", "cp1252"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired:
            return repaired
    return text


class ActionQueryMcDonalds(Action):

    def name(self) -> Text:
        return "action_query_mcdonalds"

    def get_knowledge_base_content(self) -> Text:
        """
        Reads both the TXT and CSV files and combines them into a single string context.
        """
        combined_context = ""

        # 1. Read Text File (General Info)
        try:
            with open("knowledge_base.txt", "r", encoding="utf-8") as f:
                text_data = f.read()
                combined_context += f"--- GENERAL FOOD SAFETY & POLICIES ---\n{text_data}\n\n"
        except FileNotFoundError:
            print("Warning: knowledge_base.txt not found.")

        # 2. Read CSV File (Nutrition Data)
        try:
            with open("knowledge_nutrition.csv", "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                csv_text = "--- NUTRITION DATA ---\n"
                for row in reader:
                    # Convert CSV row to a readable sentence for the LLM
                    item_name = row.get("Item", "Unknown Item")
                    details = ", ".join([f"{k}: {v}" for k, v in row.items() if k != "Item"])
                    csv_text += f"Item: {item_name} | Nutritional Values: [{details}]\n"

                combined_context += csv_text
        except FileNotFoundError:
            print("Warning: knowledge_nutrition.csv not found.")

        return combined_context

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 1. Get User Query
        user_query = tracker.latest_message.get("text")
        print(f"User Query: {user_query}")

        # 2. Load Data
        context_data = self.get_knowledge_base_content()

        # 3. Construct System Prompt
        system_instruction = f"""You are a helpful customer service assistant for McDonald's Hong Kong.
Use the provided Context to answer the user's question about food safety or nutrition.

Context:
{context_data}

Instructions:
- Answer in the SAME language as the user's question (Traditional Chinese or English). If Chinese is used, always reply in Traditional Chinese (繁體中文).
- If asked about nutrition (calories, protein, etc.), look at the 'NUTRITION DATA' section.
- If asked about safety, allergens, or policies, look at the 'GENERAL FOOD SAFETY & POLICIES' section.
- If the answer is not in the context, politely say you don't have that specific information.
- Keep answers concise and friendly."""

        # 4. Select Online (NVIDIA Free API) vs Local (LM Studio Bionic) Connection
        if LLM_MODE.lower() == "online":
            api_url = NVIDIA_API_URL
            model_name = ONLINE_MODEL_NAME
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY.strip()}",
                "Content-Type": "application/json"
            }
        else:
            api_url = LM_STUDIO_API_URL
            model_name = LOCAL_MODEL_NAME
            headers = {
                "Content-Type": "application/json"
            }

        # 5. Prepare OpenAI-Compatible Payload (works for BOTH NVIDIA NIM & LM Studio Bionic!)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            "max_tokens": 600,
            "temperature": 0.2,  # Low temperature for accurate data retrieval
            "stream": True       # Print tokens live in this Action Server terminal
        }

        # 6. Call Selected LLM API Server (stream tokens to this terminal)
        try:
            print(f"Sending request via [{LLM_MODE.upper()}] mode to {model_name} ({api_url})...")
            print("LLM Response: ", end="", flush=True)
            pieces = []

            with requests.post(
                api_url, headers=headers, json=payload, timeout=LLM_TIMEOUT, stream=True
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=False):
                    line = _as_utf8_text(raw_line).strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token = delta.get("content") or ""
                    if token:
                        token = _repair_utf8_mojibake(token)
                        print(token, end="", flush=True)
                        pieces.append(token)

            print()
            generated_text = _repair_utf8_mojibake("".join(pieces))

            # Strip reasoning/thinking blocks before the reply is sent to rasa shell
            generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL)
            generated_text = re.sub(r"<\|channel>thought.*?<channel\|>", "", generated_text, flags=re.DOTALL)
            generated_text = generated_text.strip()

            dispatcher.utter_message(text=generated_text)

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to LLM ({LLM_MODE} - {model_name}): {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Server error details: {e.response.text}")
            dispatcher.utter_message(
                text="Sorry, I am having trouble connecting to the AI language model right now. "
                     "(If using NVIDIA Online API, check your API key or model availability; "
                     "if using Local mode, ensure LM Studio Bionic Server is running on port 1234.)"
            )

        return []
