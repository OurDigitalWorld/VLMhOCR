# VLMhOCR
Scripts for VLM use with hOCR outputs

These scripts are referenced in  [this Google Colab notebook](https://colab.research.google.com/drive/1wT5sdPXNCF-EfqIw_QevhRptK1pIaj2g?usp=sharing).
The basic use can be seen in the top comments on each script. Our approach is to use 
[Eynollah](https://github.com/qurator-spk/eynollah) to segment newspaper images into PAGE-XML,
which is then used to produce paragraph-level images via the _eyPage.py_ script. Those images
can be run against [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR) via the 
_hunOcrRun.py_ script (_hunCpuRun.py_ for CPU-only). The resulting text files are converted to hOCR with the
_coords2hOCR.py_ script. Note that [Code Carbon](https://codecarbon.io/) is used help
track overhead. The final goal is a hybrid environment where 
[Tesseract](https://github.com/tesseract-ocr) is used for as much as the recognition process
as possible (due to its smaller footprint), and a VLM/LLM layer is applied selectively via
vLLM and straming. _vllmHunOcrStream.py_ is an example of using streaming.
