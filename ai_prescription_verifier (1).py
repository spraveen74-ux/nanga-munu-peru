# ai_prescription_verifier.py
# AI Medical Prescription Verification using IBM Watson & Hugging Face
# Author: sme_editz

from transformers import pipeline
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions

print("Loading Hugging Face model...")
nlp = pipeline("ner", model="dslim/bert-base-NER")

# --- IBM Watson Setup ---
IBM_APIKEY = "your_ibm_watson_api_key"
IBM_URL = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com"

authenticator = IAMAuthenticator(IBM_APIKEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2023-08-01',
    authenticator=authenticator
)
nlu.set_service_url(IBM_URL)

# --- Input Prescription Text ---
print("\nEnter the prescription text below:")
prescription_text = input("👉 ")

# --- Step 1: Hugging Face NER ---
print("\n🔹 Hugging Face NER Results:")
entities = nlp(prescription_text)
for ent in entities:
    print(f"{ent['word']} --> {ent['entity_group']} ({ent['score']:.2f})")

# --- Step 2: IBM Watson Entity Analysis ---
print("\n🔹 IBM Watson Entity Analysis:")
response = nlu.analyze(
    text=prescription_text,
    features=Features(entities=EntitiesOptions(limit=5))
).get_result()

for entity in response['entities']:
    print(f"{entity['text']} ({entity['type']}) | Relevance: {entity['relevance']:.2f}")

# --- Step 3: Basic Verification ---
print("\n🔹 Verification Check:")
required_keywords = ["mg", "tablet", "once", "daily", "patient"]
missing = [word for word in required_keywords if word.lower() not in prescription_text.lower()]

if missing:
    print("⚠️ Missing important info:", ", ".join(missing))
else:
    print("✅ Prescription seems complete and valid (basic check).")

print("\n✅ Analysis Completed.")
