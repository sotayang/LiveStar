<div align="center">

# **LiveStar & LiveStarPro: Live Streaming Assistant for Real-World Online Video Understanding**

[\[📖 LiveStarPro Paper\]](https://arxiv.org/abs/2606.17798) [\[📖 LiveStar Paper (NeurIPS'25)\]](https://arxiv.org/abs/2511.05299) [\[🤖 HF Model\]](https://huggingface.co/yzy666/LiveStar_8B) [\[🤗 HF Dataset\]](https://huggingface.co/datasets/yzy666/OmniStar-RNG) [\[🎬 Base Model\]](https://huggingface.co/yzy666/LiveStar_InternVideo_8B) [\[📄 中文解读\]](https://mp.weixin.qq.com/s/GdkpgCxrAlbVrN6AAQn74A) 

</div>

This is the **code repository** for ***LiveStar*** (NeurIPS 2025) and its journal extension ***LiveStarPro***. It provides the code, data, and pipeline for both models: LiveStar for real-time online video understanding, and LiveStarPro, which extends it with hierarchical memory for long-horizon streams. Both models can be run from the same inference harness for direct comparison (see [LiveStarPro](#livestarpro-)). 🚀🚀🚀


## News & Updates 🚀  
- `2026-07-02`:  
  🔥 **LiveStarPro Released**: The journal extension **LiveStarPro** is now available ([arXiv](https://arxiv.org/abs/2606.17798)). It adds **Tree-Structured Hierarchical Memory (TSHM)** for long-horizon streams and a **Streaming KV Cache** for faster inference — both at inference time on the same `LiveStar-8B` weights. Run it with the `--pro` flag (see [LiveStarPro](#livestarpro-)).
- `2025-09-19`:  
  🔥 **Paper Accepted**: Our work has been accepted to NeurIPS 2025! The [arXiv](https://arxiv.org/abs/2511.05299) version is now available — try our LiveStar model for streaming inference today!
- `2025-05-27`:  
  🔥 **LiveStar Released**: We've launched the [LiveStar-8B model](https://huggingface.co/yzy666/LiveStar_8B) on Hugging Face for immediate online inference!  
  - **Current Features**:  
    ✔️ Full model weights accessible  
    ✔️ Basic inference pipeline integration  
  - **Coming Soon**:  
    📅 **OmniStar Dataset**: Full release pending completion of the peer-review process  
    ⚙️ **Extended Tools**: Enhanced training scripts and evaluation protocols  

## **Overview**

**Illustration of online video understanding.** (a) Taking the RNG task as an example, online video understanding requires Video-LLMs to handle continuous streams and output at appropriate times; (b) Existing methods overly rely on learning the EOS token, leading to poor inference performance; (c)-(e) LiveStar establishes an effective response-silence training and inference framework by SCAM and SVeD without compromising basic video understanding capabilities.

![overview](./assets/images/overview.png)

### **Abstract**

Despite significant progress in Video Large Language Models (Video-LLMs) for offline video understanding, existing online Video-LLMs typically struggle to simultaneously process continuous frame-by-frame inputs and determine optimal response timing, often compromising real-time responsiveness and narrative coherence. To address these limitations, we introduce LiveStar, a pioneering live streaming assistant that achieves always-on proactive responses through adaptive streaming decoding. Specifically, LiveStar incorporates: (1) a training strategy enabling incremental video-language alignment for variable-length video streams, preserving temporal consistency across dynamically evolving frame sequences; (2) a response-silence decoding framework that determines optimal proactive response timing via a single forward pass verification; (3) memory-aware acceleration via peak-end memory compression for online inference on 10+ minute videos, combined with streaming key-value cache to achieve 1.53× faster inference. We also construct an OmniStar dataset, a comprehensive dataset for training and benchmarking that encompasses 15 diverse real-world scenarios and 5 evaluation tasks for online video understanding. Extensive experiments across three benchmarks demonstrate LiveStar's state-of-the-art performance, achieving an average 19.5% improvement in semantic correctness with 18.1% reduced timing difference compared to existing online Video-LLMs, while improving FPS by 12.0% across all five OmniStar tasks.

## **Getting Started**

This guide provides step-by-step instructions to set up the LiveStar framework, including environment configuration, model acquisition, and dataset preparation. Current implementations focus on inference capabilities with partial resource availability.

### **Installation**

1. Clone the repository

```sh
git clone https://github.com/sotayang/LiveStar.git
cd LiveStar
```

2. Install Python dependencies. Requirements: Python >= 3.9, and an NVIDIA GPU whose driver supports CUDA 12.x.

> **Note:** `flash-attn` must be installed **after** PyTorch and needs a CUDA toolkit (`nvcc`) at build time. It is therefore installed as a separate step below and is intentionally **not** listed in `requirements.txt` (otherwise `pip install -r requirements.txt` fails with `ModuleNotFoundError: No module named 'torch'`).

**Step 1 - create the environment and install PyTorch first:**
```bash
conda create -n LiveStar -y python=3.9.21
conda activate LiveStar
pip install torch==2.5.1 torchvision==0.20.1
```

**Step 2 - install the remaining dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 - install flash-attn (requires a CUDA toolkit / `nvcc`):**
```bash
# If nvcc is not already on your machine, install a CUDA toolkit, e.g. via conda:
conda install -y -c nvidia cuda-toolkit=12.4
export CUDA_HOME=$CONDA_PREFIX   # must point to the env root, NOT targets/x86_64-linux

# Build flash-attn from source. FLASH_ATTENTION_FORCE_BUILD=TRUE skips the prebuilt
# wheel download from GitHub (needed on machines without direct GitHub access).
# Optional: FLASH_ATTN_CUDA_ARCHS limits the build to your GPU's compute capability,
# which cuts build time drastically (e.g. 80 for A100/A800, 90 for H100; ";"-separated
# for multiple). Omit it to build for all archs (80;90;100;120) - much slower.
MAX_JOBS=32 FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 pip install flash_attn==2.7.4.post1 --no-build-isolation
```

### **Model Acquisition**

> **Tip:** if `huggingface.co` is slow or unreachable, prefix the download commands with `HF_ENDPOINT=https://hf-mirror.com` to use the mirror.

1. Download Fine-Tuned LiveStar Model (Recommended):

(1) Download the LiveStar-8B model from Hugging Face:

```Bash
hf download yzy666/LiveStar_8B --local-dir ./LiveStar_8B
```

(2) Move model weights to the inference directory:

```Bash
mv LiveStar_8B/*.safetensors inference/
```

2. SFT Training from Scratch (Advanced):

(1) Download the base pre-trained model:

```bash
hf download yzy666/LiveStar_InternVideo_8B --local-dir ./LiveStar_InternVideo_8B
```

(2) Prepare weights for fine-tuning:
```bash
mv LiveStar_InternVideo_8B/*.safetensors inference/
```

### **Data Preparation**

(1) Download the OmniStar dataset from Hugging Face:

```bash
hf download yzy666/OmniStar-RNG --local-dir ./OmniStar-RNG --repo-type=dataset
```


(2)  Merge the raw video folders:

```bash
# Navigate to the dataset directory
cd OmniStar-RNG
```
Since Hugging Face repositories have a limit of 10,000 files per folder, the videos are split into two directories: `videos` (9,995 files) and `videos_2` (142 files), totaling 10,137 files.

```bash
mv videos_2/* videos/
```

**Note:** Steps (3)-(5) are **deprecated** as the extracted video files are already available in the `videos` directory.
<details><summary>Deprecated Steps (3)-(5) - Click to view</summary>
  (3) Concatenate the split files:

  Use the cat command to concatenate all the split files into a single file. The split files are named from allVideos.part_aa to allVideos.part_ch, you can use the following command:

  ```Bash
  cat allVideos_tar_sep/allVideos.part_* > allVideo.tar.gz
  ```

  (4) Verify the integrity of the file (optional):

  Use the md5sum command to compute the checksum of the concatenated file and compare it with the provided checksum 43d6777701f8bfbfcc7854304245cc2c:

  ```Bash
  md5sum allVideo.tar.gz
  ```

  The output should look like this:

  ```Bash
  43d6777701f8bfbfcc7854304245cc2c  allVideo.tar.gz
  ```

  If the checksum matches 43d6777701f8bfbfcc7854304245cc2c, the file is intact and correct.

  (5) Extract the concatenated file:

  Use the tar command to extract the contents of allVideo.tar.gz:

  ```Bash
  tar -xzvf allVideo.tar.gz
  ```
</details>

(6) Extract frames from videos by running the following command:

```Bash
python utils/extract_video_frame.py --data_dir ./videos --output_dir ./video_frames
```


After completing these steps, you should see the extracted video and frame files in the OmniStar-RNG directory.

## **Inference**

![SVeD](./assets/images/SVeD.png)

To run an inference with the LiveStar model, follow these steps:

(1) Before using LiveStar for inference, ensure you have downloaded the pre-trained model weights. Then, navigate to the inference directory:
   ```bash
   cd LiveStar/inference
   ```

(2) Ensure that the model path in your script matches the actual path to the downloaded weights: `model_path = './'`. The script runs on a bundled sample video by default (`video_path = "../assets/videos/HPtIGhOsViM.mp4"`); to use your own video, edit that `video_path` line in `demo.py`.

(3) Execute the inference script using the following command:
   ```bash
   python demo.py
   ```
(4) If you want a more intuitive experience, we provide a visualization demo based on Gradio. Please run:
   ```bash
   python demo_ui.py
   ```
![Visualization](./assets/images/LiveStar_visualization.png)

## **LiveStarPro** 🌟

**LiveStarPro** is the journal extension of LiveStar for **long-horizon** online video understanding. It keeps LiveStar's SVeD inference and SCAM training unchanged and adds two inference-time modules — no retraining is needed, the **same** `LiveStar-8B` weights are used:

- **Tree-Structured Hierarchical Memory (TSHM)** — a two-tier memory for effectively unbounded streams:
  - *Short-Term Working Memory*: Peak-End compression keeps salient keyframes (low-perplexity "peaks") plus each clip's summary caption, bounding the active context regardless of stream length.
  - *Long-Term Retrieval Memory*: evicted events are organized into a **Recursive Event Tree** (semantically similar events are attached as children, with momentum-updated parent embeddings). When a response is triggered, a **hierarchical beam-descent** retrieval re-injects relevant historical event chains, with a temporal gate so only genuinely distant events are recalled.
- **Streaming KV Cache** — caches per-frame visual features and reuses the attention key/value of the stable context prefix across streaming steps, avoiding redundant recomputation of history.

### Highlights 📈

As reported in the paper, LiveStarPro sets a new state of the art for online video understanding:

- **+28.9% semantic correctness (SemCor)** and **−18.2% timing difference (TimDiff)** relative to prior online Video-LLMs — up from LiveStar's +19.5% — together with **+16.1% FPS**.
- **1.58× inference speedup** from the Streaming KV Cache versus the same model without caching, sustaining ~3 FPS on hour-long streams.
- **Long-horizon recall**: on the long (>30 min) memory-span bucket, the Recursive Event Tree reaches **37.2%** recall versus **21.3%** for a flat retrieval bank and near-chance for sliding-window baselines, while degrading far more gracefully as the memory span grows.

### Running LiveStar vs. LiveStarPro

Both modes share a single entry point, `inference/streaming_infer.py`, so a controlled comparison is one flag away:

```bash
cd LiveStar/inference

# LiveStar (base): SVeD + short-term Peak-End memory, no long-term retrieval
python streaming_infer.py --video ../assets/videos/HPtIGhOsViM.mp4

# LiveStarPro: adds the Recursive Event Tree, gated retrieval, and streaming KV cache
python streaming_infer.py --video ../assets/videos/HPtIGhOsViM.mp4 --pro
```

Common flags (run `python streaming_infer.py -h` for the full list):

| Flag | Default | Meaning |
|------|---------|---------|
| `--pro` | off | Enable the full LiveStarPro stack (TSHM tree + gated retrieval + KV cache) |
| `--l-max` | 160 | Active-context token budget that triggers Peak-End compression |
| `--alpha` | 1.06 | SVeD response-silence sensitivity (larger = fewer responses) |
| `--sigma` / `--beta` | 0.75 / 0.3 | Event-tree attach threshold / parent-embedding momentum |
| `--recall-min-gap` | 8 (with `--pro`) | Only recall events at least this many frames in the past |
| `--kv` | on with `--pro` | Reuse the verification-path streaming KV cache |
| `--trace` | off | Print per-frame SVeD / memory / retrieval decisions |

> `demo.py` remains the reference LiveStar inference script. `streaming_infer.py` reproduces LiveStar behavior without `--pro`, and layers LiveStarPro on top with `--pro`, keeping the harness identical for a fair comparison.

## **Training**

### **1. Prepare *Frame-Caption* Format Data**

(1) To fine-tune the LiveStar model, prepare your own Supervised Fine-Tuning (SFT) dataset as interleaved frame-caption sequences. Create a `.jsonl` file under the `LiveStar/datasets` directory, following the structure of `train_data.jsonl`.

(2) Next, create a meta file in JSON format under the `LiveStar/shell/data` directory. This file should provide metadata for your dataset and follow the format shown in `omnistar_train_sample.json`.


### **2. Fine-tune the Pre-trained Model**
You can fine-tune the [LiveStar-8B](https://huggingface.co/yzy666/LiveStar_8B) model directly (recommended), or start from the base [LiveStar-InternVideo-8B](https://huggingface.co/yzy666/LiveStar_InternVideo_8B) model for full SFT training. You may choose to fine-tune the model using either the full-parameter fine-tuning script or the lightweight LoRA adapter depending on your available GPU resources.


Before starting fine-tuning, make sure to set the `--meta_path` argument to the JSON meta file you created in the previous step.  

The model path in the shell scripts is set to `./inference` by default.

In the default configuration, the visual encoder is frozen to reduce memory usage. You may unfreeze it if you wish to improve performance, especially if you have sufficient computational resources.

🎈 Fine-tuning the full model typically requires 8× A800 80G GPUs.  
🎈 Fine-tuning with LoRA is much lighter and can be done with just 2× A800 80G GPUs.  

Example fine-tuning commands:
```bash
# Fine-tune the full LiveStar model with 8 GPUs (~77GB per GPU)
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/scripts/LiveStar-8B_full.sh

# Fine-tune LiveStar with LoRA on 2 GPUs (~79GB per GPU)
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/scripts/LiveStar-8B_lora.sh
```

## **OmniStar (Coming Soon)**
<details> <summary>Annotation and Evaluation</summary>

This section provides instructions for reproducing the annotation and evaluation of OmniStar.

![pipline](./assets/images/pipline_RNG.png)

### **1. Data Filtering**

Run the following commands to obtain filtered videos. 

Firstly, you should install [Open-Sora](https://github.com/hpcaitech/Open-Sora/tree/main/tools), and have a raw video dataset prepared. A meta file of the dataset information is needed for data processing. To create a meta file from a folder, run:

```Bash
python -m Data_Filtering/Open-Sora-main/tools.datasets.convert video /path_to_your_video_folder --output /path_to_save_your_meta.csv
```

Then, run the following commands to get aesthetic scores and optical flow scores of your videos. Make sure the meta file has column 'path'.

```Bash
torchrun --nproc_per_node 8 -m Data_Filtering/Open-Sora-main/tools.scoring.aesthetic.inference /path_to_save_your_meta_with_aesthetic_scores.csv --bs 1024 --num_workers 16
torchrun --standalone --nproc_per_node 8 Data_Filtering/Open-Sora-main/tools/scoring/optical_flow/inference.py /path_to_save_your_meta_with_optical_flow_scores.csv
```

With these information of videos above, you can filtering is conducted to retain only those videos containing 5 to 15 scenes,Then you can retain videos with an aesthetic score of 4 or above and with optical flow scores within the range of 0.5 to 100

### **2. Video Frame Extracting**

Video frame extraction can be directly run the following code. Run the following command:

```Bash
python utils/extract_video_frame.py --data_dir allVideo --output_dir allVideo_frame
```

![RNG](./assets/images/RNG_case.png)

</details>

## **Acknowledgment**
We would like to extend our sincere gratitude to the following projects, which were instrumental to this work:

* [InternVL](https://github.com/OpenGVLab/InternVL): For providing the powerful training codebase that served as the foundation for our implementation.

* [InternVideo](https://github.com/OpenGVLab/InternVideo): For their outstanding video foundation models, which significantly enhanced our capabilities.

## **Citation**

If you find our data useful, please consider citing our work!

```BibTeX
@article{yang2026livestarpro,
  title={LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams},
  author={Yang, Zhenyu and Zhang, Kairui and Wang, Bing and Qian, Shengsheng and Xu, Changsheng},
  journal={arXiv preprint arXiv:2606.17798},
  year={2026}
}

@article{yang2025livestar,
  title={LiveStar: Live Streaming Assistant for Real-World Online Video Understanding},
  author={Yang, Zhenyu and Zhang, Kairui and Hu, Yuhang and Wang, Bing and Qian, Shengsheng and Wen, Bin and Yang, Fan and Gao, Tingting and Dong, Weiming and Xu, Changsheng},
  journal={arXiv preprint arXiv:2511.05299},
  year={2025}
}
```
