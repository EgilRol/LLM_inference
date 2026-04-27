# LLM Inference Project

Full LLM inference pipeline in C++ and CUDA. Run ```tools/llama_3_downloader.py``` to download model weights (Llama 3 8B instruct). Run ```tools/dumper.py``` to dump model weights into binary format. Run ```make``` to build executables, and ```./bin/llm "prompt" n_tokens``` to run inference and generate ```n_tokens``` tokens.
