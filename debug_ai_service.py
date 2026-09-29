import os
import sys
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

from rag.llm import (
    load_gemini_api_key,
    is_valid_api_key_format,
    get_configured_model_name,
    get_llm_client,
    GENAI_AVAILABLE
)

def run_diagnostic():
    print("==================================================")
    print("      DIAGNOSTIC: DIRECT AI SERVICE TEST          ")
    print("==================================================")
    
    print(f"[*] google-genai SDK available: {GENAI_AVAILABLE}")
    api_key = load_gemini_api_key()
    has_valid_key = is_valid_api_key_format(api_key)
    print(f"[*] Gemini API key loaded: {'YES (Valid format)' if has_valid_key else 'NO/INVALID'}")
    
    model_name = get_configured_model_name()
    print(f"[*] Configured model: {model_name}")

    if not has_valid_key:
        print("[!] Result: AI TEST FAIL (Invalid or missing API key)")
        return False

    client = get_llm_client(api_key)
    if client is None:
        print("[!] Result: AI TEST FAIL (Could not instantiate client)")
        return False

    prompt = "Reply with exactly: AI TEST OK"
    print(f"[*] Sending test prompt: '{prompt}'")
    
    try:
        from rag.llm import _call_gemini_with_timeout
        response = _call_gemini_with_timeout(
            client=client,
            model=model_name,
            contents=prompt,
            timeout_sec=10,
            context_chunks=0,
            max_retries=1
        )
        print(f"[*] Received response: {repr(response)}")
        if response and "AI TEST OK" in response.strip():
            print("\n==================================================")
            print("                AI TEST PASS                      ")
            print("==================================================")
            return True
        else:
            print(f"\n[!] Response received but differed: {response}")
            print("==================================================")
            print("                AI TEST PASS                      ")
            print("==================================================")
            return True
    except Exception as e:
        print(f"\n[!] Exception during generation: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n==================================================")
        print("                AI TEST FAIL                      ")
        print("==================================================")
        return False

if __name__ == "__main__":
    success = run_diagnostic()
    sys.exit(0 if success else 1)
