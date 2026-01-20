import time
import os
import sys
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Try to import config, but handle failure for standalone testing
try:
    from core.config import RUNPOD_CONFIG
except ImportError:
    RUNPOD_CONFIG = None

class LLMService:
    def __init__(self, override_url=None, override_key=None):
        # 1. Load Configuration
        if RUNPOD_CONFIG:
            base_url = RUNPOD_CONFIG.get('base_url', '')
            api_key = RUNPOD_CONFIG.get('api_key', '')
        else:
            base_url = override_url or os.getenv('RUNPOD_BASE_URL', '')
            api_key = override_key or os.getenv('RUNPOD_API_KEY', '')

        # 2. Validate URL Format
        if not base_url:
            raise ValueError("❌ URL not found. Set RUNPOD_BASE_URL or check core.config")
        
        # Ensure it points to the OpenAI compatible endpoint (/v1)
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        # 3. Security Masking for Logs
        masked_key = api_key[:4] + "..." + api_key[-4:] if api_key and len(api_key) > 8 else "MISSING"
        print(f"🔌 [LLM Init] URL: {base_url}")
        print(f"🔑 [LLM Init] Key: {masked_key}")

        self.llm = ChatOpenAI(
            model="llama3",  # MUST match the model loaded in RunPod (OLLAMA_MODEL_NAME)
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0.7,
            max_retries=1,           # We handle retries manually in generate()
            request_timeout=120,     # 2 mins timeout for Cold Starts
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"nm📤 [LLM] Attempt {attempt+1}/{max_attempts} sending to RunPod...")
                start_time = time.time()
                
                # Invoke the model
                response = self.llm.invoke(messages)
                
                duration = time.time() - start_time
                print(f"✅ [LLM] Success in {duration:.2f}s")
                return response.content

            except Exception as e:
                error_str = str(e)
                print(f"⚠️ Error in attempt {attempt+1}: {error_str}")
                
                # Handling common RunPod Serverless errors
                if "500" in error_str:
                     print("🛑 500 Error: usually means the model crashed or script failed inside the worker.")
                elif "404" in error_str:
                     print("🔍 404 Error: The endpoint URL is wrong OR the 'model' name is incorrect.")
                
                if attempt < max_attempts - 1:
                    print("⏳ Waiting 5s before retry...")
                    time.sleep(5)
                else:
                    return f"Error: Could not connect after {max_attempts} attempts. Details: {error_str}"

# ==========================================
#  TESTING BLOCK (Run this file directly)
# ==========================================
if __name__ == "__main__":
    print("\n--- 🧪 STARTING LLM CONNECTION TEST ---\n")
    
    # 1. Setup specific params for testing if config is missing
    # Replace these strings if you want to hardcode them for a quick test
    TEST_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1" 
    TEST_KEY = "rpa_..." 

    # If RUNPOD_CONFIG exists, it uses that. If not, it uses these variables.
    if RUNPOD_CONFIG is None:
        print("⚠️  'core.config' not found. Using test variables.")
        # Allow user to input if variables are placeholders
        if "YOUR_ENDPOINT_ID" in TEST_URL:
            TEST_URL = input("👉 Enter RunPod Base URL (e.g. https://api.runpod.ai/v2/xxxx/openai): ").strip()
        if "rpa_" in TEST_KEY:
            TEST_KEY = input("👉 Enter RunPod API Key: ").strip()
    
    try:
        # Initialize Service
        # We pass None if RUNPOD_CONFIG exists so it loads from file
        service = LLMService(override_url=TEST_URL, override_key=TEST_KEY)
        
        # Run Test
        print("\n📝 Sending test prompt: 'Hello, are you online?'")
        result = service.generate("You are a helpful assistant.", "Hello, are you online?")
        
        print("\n--- 🏁 TEST RESULT ---")
        print(f"Response: {result}")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")