# run
- python chatbot.py

# open browser
- http://127.0.0.1:7860

# cuda
Compute Unified Device Architecture (CUDA) is a parallel computing platform and application programming interface (API) developed by NVIDIA. It allows developers to use the power of NVIDIA GPUs (graphics processing units) for general-purpose computing, including deep learning.

CUDA provides developers with the tools and functionalities needed to harness the raw computational power of NVIDIA’s GPUs. It allows developers to direct specific computing tasks to the more efficient GPU rather than the CPU. 

- nvidia-smi

# test whether cuda is available
- import  torch
- print(torch.cuda.is_available())

# download
- https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local


# references

[link] (https://www.gradio.app/guides/quickstart)
[cuda] (https://saturncloud.io/blog/what-is-assertionerror-torch-not-compiled-with-cuda-enabled/)