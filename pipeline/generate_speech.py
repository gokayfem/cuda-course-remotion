"""Generate TTS speech scripts for course slideshow videos via Gemini 2.5 Pro."""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fal_client
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_DIR = Path(__file__).resolve().parent.parent / "slideshow" / "out"
OUTPUT_DIR = Path(__file__).resolve().parent / "out"
TRANSITION_OVERLAP = 0.5  # seconds (15 frames at 30fps)
MODEL = "google/gemini-2.5-pro"
TEMPERATURE = 0.3
MAX_TOKENS = 16384
MAX_RETRIES = 3
RETRY_DELAYS = [10, 20, 40]  # exponential backoff in seconds
WORDS_PER_SECOND = 3

# ---------------------------------------------------------------------------
# Per-module data: title, durations (seconds), slide names
# ---------------------------------------------------------------------------

MODULE_DATA = {
    1: {
        "title": "GPU Architecture and First Kernel",
        "context": "This module teaches GPU architecture fundamentals and writing your first CUDA kernel.",
        "durations": [20, 35, 40, 38, 32, 35, 40, 40, 30, 35, 38, 40, 35, 38, 40, 38, 40, 36],
        "slides": [
            "Title", "WhyGPU", "CPUvsGPU", "CUDAModel", "CUDAQualifiers",
            "FirstKernel", "MemoryModel", "VectorAdd", "VectorAddFull",
            "ThreadIndexing1D", "ThreadIndexing2D", "GridStrideLoop",
            "ErrorHandling", "GPUTiming", "Exercise_SAXPY", "Exercise_ReLU",
            "Quiz", "Summary",
        ],
    },
    2: {
        "title": "Memory Hierarchy",
        "context": "This module covers the GPU memory hierarchy: global, shared, constant, and register memory, plus coalescing and bank conflicts.",
        "durations": [20, 38, 32, 42, 38, 36, 38, 40, 38, 40, 35, 35, 32, 34, 40, 40, 40, 32],
        "slides": [
            "Title", "MemoryHierarchy", "GlobalMemory", "Coalescing",
            "CoalescingCode", "SoAvsAoS", "SharedMemoryIntro", "SharedMemoryCode",
            "SharedMemTranspose", "BankConflicts", "BankConflictFix", "Registers",
            "RegisterSpilling", "ConstantMemory", "Exercise_Coalescing",
            "Exercise_SharedMem", "Quiz", "Summary",
        ],
    },
    3: {
        "title": "Thread Synchronization and Execution Model",
        "context": "This module explains warps, divergence, synchronization primitives, atomics, and cooperative groups.",
        "durations": [20, 36, 38, 40, 42, 38, 35, 40, 38, 40, 42, 36, 34, 38, 40, 40, 40, 33],
        "slides": [
            "Title", "WhatIsAWarp", "WarpScheduling", "WarpDivergence",
            "DivergencePatterns", "Syncthreads", "MemoryFences", "AtomicOps",
            "AtomicHistogram", "WarpShuffle", "WarpReduce", "WarpVote",
            "CooperativeGroups", "PracticalPatterns", "Exercise_Divergence",
            "Exercise_Atomics", "Quiz", "Summary",
        ],
    },
    4: {
        "title": "Parallel Patterns",
        "context": "This module covers fundamental parallel patterns: reduction, scan (prefix sum), histogram, and stream compaction.",
        "durations": [20, 36, 40, 38, 38, 42, 36, 42, 40, 36, 38, 36, 40, 38, 36, 40, 40, 34],
        "slides": [
            "Title", "WhyPatterns", "ReductionIntro", "ReductionV1",
            "ReductionV2", "ReductionOptimized", "ScanIntro", "ScanAlgorithms",
            "ScanCode", "HistogramIntro", "HistogramVersions", "HistogramCode",
            "CompactionIntro", "CompactionCode", "PracticalApplications",
            "Exercise_Reduction", "Quiz", "Summary",
        ],
    },
    5: {
        "title": "Performance Optimization",
        "context": "This module teaches occupancy, memory bandwidth optimization, ILP, loop unrolling, roofline model, and Nsight profiling.",
        "durations": [20, 36, 40, 38, 38, 40, 42, 40, 40, 38, 40, 36, 40, 38, 40, 40, 40, 34],
        "slides": [
            "Title", "WhyOptimize", "OccupancyIntro", "OccupancyFactors",
            "LaunchConfig", "MemoryBandwidth", "BandwidthOptimization", "ILP",
            "LoopUnrolling", "RooflineIntro", "RooflineAnalysis", "NsightIntro",
            "NsightMetrics", "ProfilingWorkflow", "CaseStudy", "Exercise",
            "Quiz", "Summary",
        ],
    },
    6: {
        "title": "Streams and Concurrency",
        "context": "This module covers CUDA streams, async transfers, pinned memory, overlap patterns, events, and multi-GPU programming.",
        "durations": [20, 36, 38, 40, 38, 42, 38, 38, 38, 36, 38, 36, 40, 36, 40, 40, 40, 34],
        "slides": [
            "Title", "WhyStreams", "StreamBasics", "AsyncTransfers",
            "PinnedMemory", "OverlapPattern", "DoubleBuffering", "EventsIntro",
            "EventTiming", "MultiGPUIntro", "MultiGPUCode", "StreamCallbacks",
            "PracticalPatterns", "CommonMistakes", "CaseStudy", "Exercise",
            "Quiz", "Summary",
        ],
    },
    7: {
        "title": "cuBLAS, cuDNN, and Libraries",
        "context": "This module introduces CUDA libraries: cuBLAS for linear algebra, cuDNN for deep learning, Thrust for parallel algorithms, and cuRAND.",
        "durations": [20, 36, 40, 38, 38, 40, 42, 38, 40, 36, 38, 36, 38, 36, 40, 40, 40, 34],
        "slides": [
            "Title", "WhyLibraries", "CublasIntro", "ColumnMajor",
            "CublasPerformance", "CudnnIntro", "CudnnAlgorithms", "ThrustIntro",
            "ThrustAlgorithms", "CurandIntro", "CurandML", "LibraryEcosystem",
            "LibraryInterop", "PracticalTips", "CaseStudy", "Exercise",
            "Quiz", "Summary",
        ],
    },
    8: {
        "title": "Matrix Multiplication Deep Dive",
        "context": "This module goes deep into matrix multiplication: naive, tiled, register-tiled, vectorized, tensor core implementations, and CUTLASS.",
        "durations": [20, 36, 40, 38, 42, 40, 42, 40, 40, 38, 38, 40, 36, 38, 38, 40, 40, 34],
        "slides": [
            "Title", "WhyMatmul", "NaiveImpl", "NaiveAnalysis",
            "TilingConcept", "TiledCode", "RegisterTiling", "VectorizedLoads",
            "OptimizationJourney", "ArithmeticIntensity", "BankConflicts",
            "TensorCores", "WhenCustom", "CUTLASS", "MLApplications",
            "Exercise", "Quiz", "Summary",
        ],
    },
    9: {
        "title": "Attention and Transformer Kernels",
        "context": "This module covers softmax, layer normalization, FlashAttention, kernel fusion, and transformer optimization techniques.",
        "durations": [20, 36, 40, 42, 40, 40, 40, 42, 42, 38, 40, 38, 40, 38, 40, 40, 40, 34],
        "slides": [
            "Title", "TransformerAnatomy", "SoftmaxIntro", "OnlineSoftmax",
            "SoftmaxCode", "LayerNormIntro", "LayerNormCode",
            "FlashAttentionIntro", "FlashAttentionAlgo", "FusionIntro",
            "FusionPatterns", "FusedBiasGelu", "FlashAttentionV2",
            "TransformerOptimizations", "CaseStudy", "Exercise", "Quiz",
            "Summary",
        ],
    },
    10: {
        "title": "Advanced Topics",
        "context": "This module covers tensor cores, mixed precision training, CUTLASS, Triton, PyTorch extensions, and career paths in GPU computing.",
        "durations": [20, 36, 40, 40, 38, 38, 40, 40, 38, 38, 40, 36, 38, 36, 42, 40, 40, 36],
        "slides": [
            "Title", "TensorCoreIntro", "WMMA", "MixedPrecisionIntro",
            "LossScaling", "AMPPattern", "CUTLASSIntro", "TritonIntro",
            "TritonVsCuda", "PyTorchExtIntro", "PyTorchExtCode",
            "LandscapeOverview", "CareerPath", "ResourcesGuide",
            "CurriculumReview", "Exercise", "Quiz", "Summary",
        ],
    },
}


# ---------------------------------------------------------------------------
# Environment & client
# ---------------------------------------------------------------------------


def load_env():
    """Load FAL_KEY from .env file."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    fal_key = os.getenv("FAL_KEY")
    if not fal_key or fal_key.startswith("<"):
        print(f"ERROR: Set a valid FAL_KEY in {env_path}")
        sys.exit(1)
    return fal_key


def make_client(fal_key):
    """Create OpenAI client pointed at fal.ai OpenRouter."""
    return OpenAI(
        api_key="placeholder",
        base_url="https://fal.run/openrouter/router/openai/v1",
        default_headers={"Authorization": f"Key {fal_key}"},
    )


# ---------------------------------------------------------------------------
# Slide timing computation
# ---------------------------------------------------------------------------


def compute_slide_timings(durations):
    """Compute start/end times accounting for 0.5s transition overlap."""
    times = []
    current = 0.0
    for d in durations:
        start = current
        end = start + d
        times.append({"start": round(start, 1), "end": round(end, 1), "duration": d})
        current = round(end - TRANSITION_OVERLAP, 1)
    return times


def format_timestamp(seconds):
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(module_num):
    """Assemble the full prompt for Gemini: role, rules, timing table, schema."""
    mod = MODULE_DATA[module_num]
    timings = compute_slide_timings(mod["durations"])

    # --- Section A: Role & Rules ---
    role_rules = """You are a professional technical narrator for the "CUDA Mastery" video course.
You are watching the actual video for this module. Generate speech narration for EVERY slide.

RULES (CRITICAL — follow exactly):

1. NEVER include math notation. Say equations in plain English.
   Example: say "Q times K transposed divided by the square root of d" NOT "QK^T/√d"

2. Use abbreviations and acronyms naturally as spoken words — do NOT spell them out letter by letter.
   Say "GPU" not "G P U", "CUDA" not "C U D A", "CPU" not "C P U", etc.
   Common terms: GPU, CPU, CUDA, LLM, FP16, FP32, SAXPY, ReLU, TFLOPS,
   cuBLAS, cuDNN, WMMA, CUTLASS, ILP, SoA, AoS, SRAM, DRAM, HBM, SM,
   SIMT, PTX, GEMM, AMP, GeLU, MLP, API, RAM, ALU — all spoken naturally.

3. Use a friendly, encouraging teacher tone — NOT a textbook.

4. Reference visual elements naturally: "as you can see here", "notice the diagram",
   "look at the code on screen", "the animation shows".

5. Budget approximately 3 words per second per slide (keeps TTS pacing natural).

6. Cover ALL content shown on each slide — do not skip information.

7. Do NOT say slide numbers aloud. Never say "Slide 1" or "on slide three".

8. For code slides, walk through the key lines and explain what they do.

9. For quiz slides, read each question, pause, then reveal the answer.

10. For exercise slides, explain the task and give encouraging hints.

11. NEVER use backticks, underscores, or code formatting in the speech text.
   This text will be read aloud by a TTS model, so it must be plain spoken language.
   Say "cuda malloc" not "`cudaMalloc`", say "global qualifier" not "`__global__`",
   say "thread index dot x" not "`threadIdx.x`", say "cuda mem copy" not "`cudaMemcpy`".
   Write everything as natural spoken words with no special characters."""

    # --- Section B: Timing Table ---
    timing_lines = ["SLIDE TIMING:"]
    for i, (slide_name, timing) in enumerate(zip(mod["slides"], timings)):
        word_budget = int(timing["duration"] * WORDS_PER_SECOND)
        timing_lines.append(
            f"Slide {i} | {slide_name:30s} | "
            f"{format_timestamp(timing['start'])} - {format_timestamp(timing['end'])} | "
            f"{timing['duration']}s | ~{word_budget} words"
        )
    timing_table = "\n".join(timing_lines)

    # --- Section C: Module context ---
    module_context = f"MODULE {module_num}: {mod['title']}\n{mod['context']}"

    # --- Section D: Output format ---
    output_format = """OUTPUT FORMAT:
Return ONLY valid JSON matching this exact schema (no markdown fences, no extra text):

{
  "segments": [
    {
      "slide_index": 0,
      "slide_name": "Title",
      "start_time": "00:00",
      "end_time": "00:20",
      "duration_seconds": 20,
      "speech": "Welcome to C U D A Mastery..."
    }
  ]
}

IMPORTANT: Return ONLY the JSON object. No ```json fences. No explanation before or after.
There must be exactly """ + str(len(mod["slides"])) + """ segments, one per slide."""

    return f"{role_rules}\n\n{timing_table}\n\n{module_context}\n\n{output_format}"


# ---------------------------------------------------------------------------
# Video upload
# ---------------------------------------------------------------------------


def upload_video(module_num):
    """Upload module video to fal CDN, return URL."""
    video_path = VIDEO_DIR / f"module{module_num}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"  Uploading {video_path.name} ...")
    for attempt in range(MAX_RETRIES):
        try:
            url = fal_client.upload_file(video_path)
            print(f"  Upload complete: {url[:80]}...")
            return url
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  Upload failed (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Upload failed after {MAX_RETRIES} attempts: {e}") from e


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------


def call_gemini(client, video_url, prompt):
    """Send video + prompt to Gemini 2.5 Pro, return raw response text."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(raw, expected_segments):
    """Strip markdown fences if present, parse JSON, validate segment count."""
    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    data = json.loads(text)

    segments = data.get("segments", [])
    if len(segments) != expected_segments:
        raise ValueError(
            f"Expected {expected_segments} segments, got {len(segments)}"
        )

    return data


# ---------------------------------------------------------------------------
# Per-module pipeline
# ---------------------------------------------------------------------------


def process_module(module_num, client, dry_run=False):
    """Full pipeline for one module: upload, prompt, call, parse, save."""
    mod = MODULE_DATA[module_num]
    num_slides = len(mod["slides"])
    timings = compute_slide_timings(mod["durations"])
    total_duration = timings[-1]["end"]

    print(f"\n{'='*60}")
    print(f"Module {module_num}: {mod['title']}")
    print(f"  {num_slides} slides, {total_duration:.1f}s total")
    print(f"{'='*60}")

    prompt = build_prompt(module_num)

    if dry_run:
        print("\n--- DRY RUN: Prompt ---")
        print(prompt)
        print("--- END ---\n")
        return True

    # Upload video
    video_url = upload_video(module_num)

    # Call Gemini with retry
    raw_response = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Calling Gemini 2.5 Pro (attempt {attempt + 1})...")
            raw_response = call_gemini(client, video_url, prompt)
            print(f"  Response received ({len(raw_response)} chars)")
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  API error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  API failed after {MAX_RETRIES} attempts: {e}")
                _save_raw_fallback(module_num, str(e))
                return False

    # Parse response
    try:
        data = parse_response(raw_response, num_slides)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Parse error: {e}")
        print("  Retrying with strict JSON instruction...")
        try:
            retry_prompt = prompt + "\n\nYour previous response was not valid JSON. Return ONLY the JSON object, nothing else."
            raw_response = call_gemini(client, video_url, retry_prompt)
            data = parse_response(raw_response, num_slides)
        except Exception as retry_err:
            print(f"  Parse retry failed: {retry_err}")
            _save_raw_fallback(module_num, raw_response)
            return False

    # Build final output
    output = {
        "module": module_num,
        "title": mod["title"],
        "total_duration_seconds": total_duration,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "segments": data["segments"],
    }

    # Save
    output_path = OUTPUT_DIR / f"module{module_num}_speech.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")
    return True


def _save_raw_fallback(module_num, content):
    """Save raw/error response for debugging."""
    fallback_path = OUTPUT_DIR / f"module{module_num}_speech.raw.txt"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fallback_path, "w") as f:
        f.write(content)
    print(f"  Raw response saved: {fallback_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS speech scripts for CUDA Mastery videos via Gemini 2.5 Pro"
    )
    parser.add_argument(
        "--modules",
        type=int,
        nargs="+",
        choices=range(1, 11),
        metavar="N",
        help="Module numbers to process (default: all 1-10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without making API calls",
    )
    args = parser.parse_args()

    modules = args.modules or list(range(1, 11))

    if not args.dry_run:
        fal_key = load_env()
        client = make_client(fal_key)
    else:
        client = None

    print(f"Processing modules: {modules}")
    print(f"Video dir: {VIDEO_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    results = {}
    for mod_num in modules:
        success = process_module(mod_num, client, dry_run=args.dry_run)
        results[mod_num] = success

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mod_num, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  Module {mod_num:2d}: {status}")

    failed = [m for m, s in results.items() if not s]
    if failed:
        print(f"\nFailed modules: {failed}")
        print(f"Retry with: python3 generate_speech.py --modules {' '.join(str(m) for m in failed)}")
        sys.exit(1)

    print("\nAll done!")


if __name__ == "__main__":
    main()
