"""Load base PolyGloss (no adaptation) and gloss one sentence.
 
    pip install torch "transformers>=4.51"
    python3 src/scritps/polygloss_demo.py
"""
import torch
from transformers import AutoModelForSeq2SeqLM
 
MODEL_ID = "lecslab/polygloss-byt5-interleaved-2025-12-28"
 
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
 
# trust_remote_code: the model repo supplies predict_igt() (polygloss_seq2seq.py).
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
 
# Predict IGT outputs
# transcription: Uspanteko, Mayan language in Guatemala
# transcription = "o sey xtok rixoqiil"
# translation = "O sea busca esposa."
# lang = "Uspanteco"
# metalang = "Spanish"

# Ex. Tiriki, un-adapted
transcription = "vamunaka vusela"
translation = "She had prepared porridge"
lang = "Tiriki"
metalang = "English"

output = model.predict_igt(transcription, translation, lang, metalang)
print(output)
 