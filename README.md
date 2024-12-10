# SealAI's Stable DiffusionCpp

Inference Implementation for stable diffusion. Powered by SealAI's acceleration engine.

**See our tech report here: https://arxiv.org/pdf/2412.05781**

----
Stable diffusion plays a crucial role in generating high-quality images. However,
image generation is time-consuming and memory-intensive. To address this, stable-
diffusion.cpp (Sdcpp) emerges as an efficient inference framework to accelerate
the diffusion models. Although it is lightweight, the current implementation of
ggml_conv_2d operator in Sdcpp is suboptimal, exhibiting both high inference
latency and massive memory usage. 

Our framework delivers correct end-to-end results across various stable diffusion
models, including SDv1.4, v1.5, v2.1, SDXL, and SDXL-Turbo. Our evaluation
results demonstrate a speedup up to **2.76×** for individual convolution layers and an
inference speedup up to **4.79×** for the overall image generation process, compared
with the original Sdcpp.


| Model    | Steps | Image Size | Type | Our Acceleration |
|----------|-------|------------|------|-------------|
| Sdxl     | 20    | 1024×1024  | F32  | 4.79×       |
|          |       |            | F16  | 3.06×       |
| Sd2      | 20    | 768×768    | F32  | 2.02×       |
|          |       |            | F16  | 1.68×       |
| Sd1.5    | 20    | 512×512    | F32  | 1.84×       |
|          |       |            | F16  | 1.51×       |
##### Latency Comparison with Sdcpp  on M1 Pro (16GB Memory and macOS 15.1)
