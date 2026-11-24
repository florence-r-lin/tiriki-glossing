# usage: CUDA_VISIBLE_DEVICES=0 python3 test.py
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "lecslab/polygloss-byt5-interleaved-2025-12-28"
adapter_dir = "checkpoints/lora_r8"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_dir).to("cuda").eval()

# Uses template from polygloss.interleaved.t2sg.prompt
template = (
    "Predict the glosses and morphological segmentation (in parentheses) for the following text in Tiriki.\n\n"
    "Text in Tiriki: {transcription}\n"
    "Translation in English: {translation}\n\n"
    "Output: "
)

# prompts transcription then translation. Enter on transcription to quit.
while True:
    transcription = input("\nTiriki (blank to quit): ").strip()
    if not transcription:
        break
    translation = input("English: ").strip()

    prompt = template.format(transcription=transcription, translation=translation)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, num_beams=2, max_new_tokens=1024)
    print("PREDICTED:", tokenizer.decode(output[0], skip_special_tokens=True))