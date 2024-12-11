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



## How to run


### Build

```shell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Run

```
usage: ./bin/sd [arguments]

arguments:
  -h, --help                         show this help message and exit
  -M, --mode [MODEL]                 run mode (txt2img or img2img or convert, default: txt2img)
  -t, --threads N                    number of threads to use during computation (default: -1).
                                     If threads <= 0, then threads will be set to the number of CPU physical cores
  -m, --model [MODEL]                path to model
  --vae [VAE]                        path to vae
  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
  --control-net [CONTROL_PATH]       path to control net model
  --embd-dir [EMBEDDING_PATH]        path to embeddings.
  --stacked-id-embd-dir [DIR]        path to PHOTOMAKER stacked id embeddings.
  --input-id-images-dir [DIR]        path to PHOTOMAKER input id images dir.
  --normalize-input                  normalize PHOTOMAKER input id images
  --upscale-model [ESRGAN_PATH]      path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now.
  --upscale-repeats                  Run the ESRGAN upscaler this many times (default 1)
  --type [TYPE]                      weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)
                                     If not specified, the default is the type of the weight file.
  --lora-model-dir [DIR]             lora model directory
  -i, --init-img [IMAGE]             path to the input image, required by img2img
  --control-image [IMAGE]            path to image condition, control net
  -o, --output OUTPUT                path to write result image to (default: ./output.png)
  -p, --prompt [PROMPT]              the prompt to render
  -n, --negative-prompt PROMPT       the negative prompt (default: "")
  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)
  --strength STRENGTH                strength for noising/unnoising (default: 0.75)
  --style-ratio STYLE-RATIO          strength for keeping input identity (default: 20%)
  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)
                                     1.0 corresponds to full destruction of information in init image
  -H, --height H                     image height, in pixel space (default: 512)
  -W, --width W                      image width, in pixel space (default: 512)
  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}
                                     sampling method (default: "euler_a")
  --steps  STEPS                     number of sample steps (default: 20)
  --rng {std_default, cuda}          RNG (default: cuda)
  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)
  -b, --batch-count COUNT            number of images to generate.
  --schedule {discrete, karras}      Denoiser sigma schedule (default: discrete)
  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
  --vae-tiling                       process vae in tiles to reduce memory usage
  --control-net-cpu                  keep controlnet in cpu (for low vram)
  --canny                            apply canny preprocessor (edge detection)
  -v, --verbose                      print extra info
```

### txt2img example


```sh
./bin/sd -m ../models/sd-v1-4.ckpt -p "a lovely cat"
# ./bin/sd -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat"
# ./bin/sd -m ../models/sd_xl_base_1.0.safetensors --vae ../models/sdxl_vae-fp16-fix.safetensors -H 1024 -W 1024 -p "a lovely cat" -v
```

## How to cite us

```
@misc{ng2024opensourceaccelerationstablediffusioncpp,
      title={Open-Source Acceleration of Stable-Diffusion.cpp}, 
      author={Jingxu Ng and Cheng Lv and Pu Zhao and Wei Niu and Juyi Lin and Minzhou Pan and Yun Liang and Yanzhi Wang},
      year={2024},
      eprint={2412.05781},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05781}, 
}
```

