"""
hunOcrRun.py - run HunyuanOCR to loop through image set

This is a sample from the distribution, just some 
customizations to:

    1) work with an aging A2 card
    2) loop through a set of images 
    3) use codecarbon to measure overhead

- art rhyno, u. of windsor & ourdigitalworld
"""

from codecarbon import EmissionsTracker
import argparse,glob,os,sys

from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor

# Load the model
model_path = "tencent/HunyuanOCR"

from vllm import LLM

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

# with a more capable GPU card then this can be as simple as this:
#llm = LLM(model=model_path, trust_remote_code=True)

# for my aging A2, this probaby slows the process down

llm = LLM(
    model=model_path,
    max_num_batched_tokens=4096,  # Lowered from 8192 to prevent token-allocation crashes
    max_model_len=4096,           # Lowered to reduce KV cache size constraints
    kv_cache_dtype="fp8",
    max_num_seqs=1,
    enable_chunked_prefill=True,
    gpu_memory_utilization=0.75,  # Lowered from 0.88 to give the A2 ample cushion
    enable_prefix_caching=False,
    mm_processor_cache_gb=0.0,
    enforce_eager=True            # FORCED: Stops high-memory CUDA Graph compilation spikes
)


processor = AutoProcessor.from_pretrained(model_path)

# Set inference parameters
sampling_params = SamplingParams(temperature=0, max_tokens=16384)

# sample provide prompts
#        {"type": "text", "text": "检测并识别图片中的文字，将文本坐标格式化输出。" } # "Detect and recognize text, output with coordinates"
#        {"type": "text", "text": "Detect and recognize text, output with coordinates." } # "Detect and recognize text, output with coordinates"
#        {"type": "text", "text": "Detect and recognize text in the image, and output the text coordinates in a formatted manner."}
# Prepare prompt for detection
#        {"type": "text", "text": "Detect and recognize text, output with coordinates." } 
#        {"type": "text", "text": "Detect and recognize text, output with coordinates." } 
# {"type": "text", "text": "Detect and recognize text, output without coordinates." } 

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

