"""
vllmHunOcrStream.py - streaming client example

This is a quick proof-of-concept of a client
application using vLLM streaming.

- art rhyno, u. of windsor & ourdigitalworld
"""

import base64
import time
from openai import OpenAI
from codecarbon import EmissionsTracker


client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

IMAGE_PATH = "document.png"
import re

""" parseHunyuanOcr - use regex to pull out coordinates """
def parseHunyuanOcr(text: str):
    #with help from Google Gemini
    pattern = r"<｜hy_place▁holder▁no▁112｜>(.*?)<｜hy_place▁holder▁no▁113｜><｜hy_place▁holder▁no▁110｜>\((.*?)\),\((.*?)\)<｜hy_place▁holder▁no▁111｜>"
    matches = re.findall(pattern, text)

    parsed_results = []
    for text_content, pt1, pt2 in matches:
        ymin, xmin = map(int, pt1.split(','))
        ymax, xmax = map(int, pt2.split(','))

        parsed_results.append({
            "text": text_content.strip(),
            "box_2d": [ymin, xmin, ymax, xmax]
        })
    return parsed_results


""" encodeLocalImage - use base64 encoding to prepare image """
def encodeLocalImage(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

try:
    tracker = EmissionsTracker()
    tracker.start()
    base64_image = encodeLocalImage(IMAGE_PATH)

    start_time = time.perf_counter()

    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                },
                {
                    "type": "text",
                    "text": "Detect and recognize text in the image, and output the word-level text coordinates in a formatted manner."
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="tencent/HunyuanOCR",
        messages=messages,
        temperature=0.0,
        stream=True,  # Streams chunks instantly as the GPU generates them
        extra_body={
            "top_k": 1,
            "repetition_penalty": 1.0,
            "skip_special_tokens": False
        },
    )

    full_para = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_para += chunk.choices[0].delta.content

    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000

    print(f"Total Turnaround Time: {execution_time_ms / 1000:.2f} seconds")
    print("================================================\n")

    ocr_data = parseHunyuanOcr(full_para)
    print("ocr_data", ocr_data)
    emissions = tracker.stop()
    print(f"Emissions: {emissions} kg CO₂")

except Exception as e:
    print(f"Execution failed: {str(e)}")
