import os
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize key
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

MODEL = "gpt-4o"


# Pydantic Model Definition
class Rule(BaseModel):
    rule_id: str
    rule_text: str
    operator: Optional[str] = None
    rules: Optional[List["Rule"]] = None  # This handles nested rules

# Use model_rebuild()
Rule.model_rebuild()


class Guideline(BaseModel):
    title: str
    insurance_name: str
    rules: Rule


# Step 1: Fetch and Extract the Guidelines Text from the Policy/I section on the webpage
def fetch_aetna_guidelines(url: str) -> str:
    """
    Fetch the Aetna clinical policy bulletin page and extract the text from the ordered list
    that comes immediately after the <h2 class="policyHead">Policy</h2> header.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the header with class "policyHead" and text "Policy"
    header = soup.find("h2", class_="policyHead", string="Policy")
    if header:
        # Find the first ordered list (<ol>) after the header
        ol = header.find_next("ol")
        if ol:
            extracted_text = ol.get_text(separator="\n").strip()
            return extracted_text
        else:
            raise ValueError("Ordered list not found after the Policy header.")
    else:
        raise ValueError("Policy header not found.")


# Step 2: Transform the data with OpenAI
def extract_rules_with_openai(text: str, guideline_title: str, insurance_name: str) -> Guideline:
    """
    Given the extracted text, use OpenAI to generate a JSON structure that follows the
    required Pydantic schema.
    """
    # Prompt without explicit instructions for title and top-level rule_text
    prompt = f"""
        Act as an NLP expert. Extract the medical necessity guidelines from the text provided below
        and convert them into a structured JSON object that conforms exactly to the following schema:

        Schema:
        {{
            "title": string,
            "insurance_name": string,
            "rules": {{
                "rule_id": string,
                "rule_text": string,
                "operator": string or null,
                "rules": list of nested rules or null
            }}
        }}

        The text to be referenced is:
        {text}

        Output only the JSON.
    """

    # Call OpenAI API
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an NLP expert that extracts and transforms unstructured text into JSON."
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
    )

    result = response.choices[0].message.content.strip()
    # Optional: Remove markdown formatting if present (```json ... ```)
    if result.startswith("```") and result.endswith("```"):
        result = "\n".join(result.split("\n")[1:-1]).strip()

    try:
        data = json.loads(result)
        # Override values to ensure the expected output
        data["title"] = guideline_title
        data["insurance_name"] = insurance_name
        if "rules" in data:
            data["rules"]["rule_text"] = "Medical Necessity"
        # Use model_validate() instead of parse_obj()
        guideline = Guideline.model_validate(data)
        return guideline
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Error processing output from OpenAI: {e}")


# Main Execution Section
def main():
    # URL for the Aetna clinical policy bulletin page
    url = "https://www.aetna.com/cpb/medical/data/300_399/0369.html"
    try:
        # Step 1: Scrape and extract the medical necessity section
        extracted_text = fetch_aetna_guidelines(url)

        # Set default title and insurance name
        guideline_title = "Chronic Fatigue Syndrome"
        insurance_name = "Aetna"

        # Step 2: Use OpenAI to transform the text into structured JSON output
        guideline = extract_rules_with_openai(extracted_text, guideline_title, insurance_name)

        # Output the structured JSON matching the BaseModel structure
        structured_json = json.dumps(guideline.model_dump(), indent=2)
        print("Structured JSON Output:\n", structured_json)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
