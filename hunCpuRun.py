"""
hunOcrRun.py - run HunyuanOCR to loop through image set

This is a sample from the distribution, just some 
customizations to:

    1) work with CPU instead of GPU
    2) loop through a set of images 
    3) use codecarbon to measure overhead

- art rhyno, u. of windsor & ourdigitalworld
"""

from codecarbon import EmissionsTracker
import argparse,glob,os,sys

from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor

parser = argparse.ArgumentParser()
arg_named = parser.add_argument_group("named arguments")
arg_named.add_argument("-f","--folder",
                           default="imgs",
                           help="input folder")
arg_named.add_argument('-o', '--ext', type=str,
                           default="jpg",
                           help="image extension")

args = parser.parse_args()

if args.folder == None or not os.path.exists(args.folder):
        print("missing input folder")
        sys.exit()

tracker = EmissionsTracker()
tracker.start()

# 1. Bind OpenMP threads to physical cores to maximize compute layout
# Change "0-15" to match your actual number of physical CPU cores (e.g., 0-7 or 0-31)
os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-15"
os.environ["OMP_NUM_THREADS"] = "16"

# 2. Allocate cache space out of 48GB RAM pool 
# Setting this to 16 leaves ~26GB for model weights and OS tasks
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "16"

model_path = "tencent/HunyuanOCR"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},
    max_num_batched_tokens=8192, 
    max_model_len=8192,
    max_num_seqs=1,
    enable_chunked_prefill=True,
    enable_prefix_caching=False # Disabled: OCR images are unique per prompt
)

processor = AutoProcessor.from_pretrained(model_path)
# Recommended greedy sampling parameter for exact structural extraction
sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

#this is what to use for word coords
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "<image>\nDetect and recognize text in the image, and output the word-level text coordinates in a formatted manner."}
    ]}
]

# Apply chat template
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# This loops through the images extracted from Eynollah
for img_path in sorted(glob.glob(args.folder + "/*." + args.ext)):
    print("candidate image for OCR", img_path)
    img_base = img_path.split(".")[0]
    img = Image.open(img_path)
    inputs = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
    # Generate
    output = llm.generate([inputs], sampling_params)[0]
    result = output.outputs[0].text
    # change coordinate layout
    result = result.replace('),(','],[')
    result = result.replace(')',')\n')
    result = result.replace('],[','),(')
    with open("%s.txt" % img_base, "w", encoding="utf-8") as file:
        file.write(result)

emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO₂")
