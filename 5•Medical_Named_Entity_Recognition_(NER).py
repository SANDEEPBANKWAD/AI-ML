# Install Dependencies
# pip install spacy transformers torch pandas

# ***Medical NER with BioClinicalBERT-based Model
import pandas as pd
import spacy
from transformers import pipeline

# Load a medical NER model from Hugging Face
# You can replace with another biomedical NER model if desired
ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

# Example MIMIC-III clinical note
clinical_note = """
The patient has a history of diabetes mellitus and hypertension.
He complained of chest pain and shortness of breath.
Current medications include metformin, aspirin, and lisinopril.
"""

# Run NER
entities = ner_pipeline(clinical_note)

print("Extracted Medical Entities:")
for entity in entities:
    print(
        f"Entity: {entity['word']}, "
        f"Type: {entity['entity_group']}, "
        f"Score: {entity['score']:.3f}"
    )

# Batch Processing Code***
import pandas as pd
from transformers import pipeline

# Load model
ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

# Load MIMIC notes
df = pd.read_csv("mimic_notes.csv")

results = []

for idx, note in enumerate(df["TEXT"]):
    try:
        entities = ner_pipeline(str(note))

        for e in entities:
            results.append({
                "note_id": idx,
                "entity": e["word"],
                "category": e["entity_group"],
                "confidence": round(e["score"], 4)
            })

    except Exception as ex:
        print(f"Error in note {idx}: {ex}")

output_df = pd.DataFrame(results)

output_df.to_csv("medical_entities.csv", index=False)

print("NER results saved to medical_entities.csv")

# Categorizing Diseases, Symptoms, and Drugs****
diseases = []
symptoms = []
drugs = []

for ent in entities:
    label = ent["entity_group"].lower()

    if "disease" in label:
        diseases.append(ent["word"])

    elif "symptom" in label:
        symptoms.append(ent["word"])

    elif "drug" in label or "chemical" in label:
        drugs.append(ent["word"])

print("\nDiseases:", diseases)
print("Symptoms:", symptoms)
print("Drugs:", drugs)

# Using spaCy for Further Processing******
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(clinical_note)

print("\nTokens:")
for token in doc:
    print(token.text, token.pos_)

