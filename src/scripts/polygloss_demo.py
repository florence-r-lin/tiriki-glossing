"""Load base PolyGloss (no adaptation) and gloss one sentence.
 
    pip install torch "transformers>=4.51"
    python3 src/scritps/polygloss_demo.py
"""
import torch
from transformers import AutoModelForSeq2SeqLM
 
model_id = "lecslab/polygloss-byt5-interleaved-2025-12-28"
 
device = "cuda"
 
# trust_remote_code: the model repo supplies predict_igt() (polygloss_seq2seq.py).
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

# predict_igt() builds its inputs on CPU; move them to the GPU before generate() runs.
_orig_generate = model.generate
def _generate_on_device(*args, **kwargs):
    kwargs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
    return _orig_generate(*args, **kwargs)
model.generate = _generate_on_device
 
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
 