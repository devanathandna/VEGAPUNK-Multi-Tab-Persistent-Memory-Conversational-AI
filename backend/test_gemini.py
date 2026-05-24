import os
import google.generativeai as genai
from dotenv import load_dotenv

def main():
    """
    A simple command-line interface to test the Google Gemini model.
    """
    # Load environment variables from .env file
    load_dotenv()

    # --- Configuration ---
    api_key = os.getenv("GEMINI_API_KEY")
    #api_key = 'AIzaSyC7sGr4OijZatfWVfm6yY8ZOjXz0W7O7jw'
    api_key = 'AIzaSyBl3bDg7z009wutF_p5Atei1Qnp3dsZh_o'
    model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
    model_name = 'gemini-2.5-flash'

    # Check if API key is available
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file.")
        print("Please add your API key to the .env file.")
        return

    # --- Initialize Model ---
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini model '{model_name}' initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing Gemini model: {e}")
        return

    # --- Interactive Chat Loop ---
    print("\n🤖 VEGAPUNK Gemini CLI Tester")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 40)

    while True:
        try:
            # Get user input
            prompt = input("You: ")

            # Check for exit command
            if prompt.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break

            # --- Generate Response ---
            if not prompt:
                continue
                
            print("\n🧠 Thinking...")
            response = model.generate_content(prompt)
            
            # Print the response
            print(f"\nGemini: {response.text}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\n👋 Session ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
