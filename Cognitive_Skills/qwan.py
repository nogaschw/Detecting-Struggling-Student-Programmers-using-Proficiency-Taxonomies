import gc
import re
import json
import torch
import pandas as pd
from Config import Config
config = Config()

# Import from unsloth
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats
from huggingface_hub import hf_hub_download


#########################
# 1. Load the QWAN Model
#########################
# IMPORTANT: Replace with your actual model path if using a local fine-tuned version
FINETUNED_MODEL_PATH = "Qwen/Qwen1.5-7B-Chat"
torch.cuda.empty_cache()

print("Loading the QWAN model...")
model, tokenizer = FastModel.from_pretrained(
    model_name = FINETUNED_MODEL_PATH,
    max_seq_length = 1024,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False
    # token = "hf_..."  # If gated
)

model.eval()


#########################
# 2. Load the Test Data
########################
data = 1
snapshot = 0
name = f"{'CW' if data == 0 else 'FC'}sn{'first' if snapshot == 0 else 'last'}QWAN"

path_df = config.path_saved_codeworkout if data == 0 else config.path_saved_falcon
questions_df = pd.read_excel(config.codeworkout_questions_path) if data == 0 else pd.read_csv(config.falconcode_questions_path)
id_question = 'ProblemID' if data == 0 else 'id'
prompt = 'Requirement' if data == 0 else 'prompt'

df = pd.read_pickle(path_df)
df = df[['student_id', 'prev_tasks_id', 'prev_tasks']] if data == 0 else df[['student_id', 'course_id', 'prev_tasks_id', 'prev_tasks']]
df = df[~df['student_id'].duplicated()]

#########################
# 3. Define the System Prompt
#########################
system_prompt = (
    "You are a computer science teacher evaluating student code. "
    "Evaluate the following programming skills on a scale from 1 to 5, where "
    "1 = weak, 2 = fair, 3 = average, 4 = strong, 5 = excellent. "
    "Provide ONLY a JSON object, no extra text, no explanation, no code block, no markdown.Format strictly like this:\n"
    "{\"Decomposition\": 3, \"Algorithmic Design\": 4, \"Reading Comprehension\": 5}"
    "Definitions:\n"
    "- Decomposition: breaking the problem into parts — 1 = no understanding, 2 = only simple parts, "
    "3 = basic decomposition, 4 = mostly complete, 5 = excellent decomposition.\n"
    "- Algorithmic Design: choosing and structuring algorithms — 1 = no algorithm or wrong approach, "
    "2 = basic approach, 3 = reasonable structure, 4 = good design, 5 = excellent design.\n"
    "- Reading Comprehension: understanding the problem description — 1 = misunderstood task, "
    "2 = partially understood, 3 = mostly understood, 4 = good understanding, 5 = excellent understanding."
)

#########################
# 4. Helper: Build Chat Template
#########################
def build_prompt(user_input):
    """
    Build prompt for Qwen1.5-Chat using role-based messages.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

if data == 0:
    df['user_input'] = df.apply(
        lambda row: [
            f'Task: {questions_df[questions_df[id_question] == int(row["prev_tasks_id"][i])][prompt].iloc[0]}\n\n'
            f'Student’s code:\n{row["prev_tasks"][i][snapshot]}'
            for i in range(len(row['prev_tasks_id']))
        ],
        axis=1
    )
else:
    # clean question
    questions_df['prompt'] = questions_df['prompt'].apply(lambda x: x.split('PROBLEM STATEMENT:')[-1] if x.__contains__("PROBLEM STATEMENT:") else x)
    questions_df['prompt'] = questions_df.prompt.apply(lambda text: re.sub(r'\bPROBLEM STATEMENT: \b', '', text).strip())
    questions_df[id_question] = questions_df[id_question].apply(lambda x: x.lower())
    df['user_input'] = df.apply(
        lambda row: [
            f"Task: {questions_df[(questions_df[id_question] == row['prev_tasks_id'][i]) & (questions_df['course_id'] == row['course_id'])][prompt].iloc[0]}\n\n"
            f"Student’s code:\n{row['prev_tasks'][i][snapshot]}"
            for i in range(len(row['prev_tasks_id']))
        ],
        axis=1
    )

def gen(batch_prompts):
    tokenized = [tokenizer(p, return_tensors="pt") for p in batch_prompts]
    prompt_lens = [t["input_ids"].shape[-1] for t in tokenized]

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.3,
            top_p=0.95,
            top_k=64,
            do_sample=True
        )

    results = []
    for out, prompt_len in zip(outputs, prompt_lens):
        text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results.append(text)

    return results


all_prompts = [build_prompt(prompt) for row in df['user_input'] for prompt in row]

print(name, f'/home/nogaschw/Codeworkout/Thesis/{name}.csv', flush=True)
#########################
# 5. Generate Predictions
#########################
batch_size = 8
all_outputs = []

for i in range(0, len(all_prompts), batch_size):
    batch = all_prompts[i:i + batch_size]
    outputs = gen(batch)  # your existing function
    all_outputs.extend(outputs)
    if (i // batch_size) % 10 == 0 and i != 0:
        print(f'{i} / {len(all_prompts)}', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

#########################
# 6. Get Scores
#########################
def extract_scores(text, i, loop_num=0):
    # Find the first JSON block in the text
    text = re.sub(r'[^\x00-\x7F]+', '', text).lower()  # removes non-ASCII
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            decomposition = data.get("decomposition")
            algorithmic = data.get("algorithmic_design") or data.get("algorithmic design")
            reading = data.get("reading comprehension") or data.get("reading_comprehension")
            return decomposition, algorithmic, reading
        except json.JSONDecodeError:
            print("Failed to parse JSON!")
            if loop_num > 2:
                return None, None, None
            return extract_scores(gen(all_prompts[i])[0], i, loop_num+1)
    else:
        print("No JSON found!")
        if loop_num > 2:
            return None, None, None
        return extract_scores(gen(all_prompts[i])[0], i, loop_num+1)
    

decomposition = []
alg = []
reading = []
for i, o in enumerate(all_outputs): 
    decomp, algo, read = extract_scores(o, i)
    decomposition.append(decomp)
    alg.append(algo)
    reading.append(read)


d = []
a = []
r = []
for i in range(0, len(decomposition), 30):
    d.append(decomposition[i:i+30])
    a.append(alg[i:i+30])
    r.append(reading[i:i+30])

df['decomposition'] = d
df['alg'] = a
df['reading'] = r

df.to_csv(f'/home/nogaschw/Codeworkout/Thesis/{name}.csv')