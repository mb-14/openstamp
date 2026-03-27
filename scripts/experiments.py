#!/usr/bin/env python3

import subprocess
import concurrent.futures
import itertools
import os
import multiprocessing

NUM_SAMPLES = 500
# ==== Common Config ====
seeds = [12997009, 22983996, 15485863]
# models = ["allenai/Olmo-3-1025-7B", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-7B", "microsoft/phi-4", "mistralai/Mistral-7B-v0.3"]
datasets = ["arxiv", "wikipedia", "booksum", "realnewslike"]
datasets = ["realnewslike"]
# Options: mb, mb_binom, noise, kgw, kgw_llr, distilled, gaussmark, rl, unigram
watermark_type = "mb"
paraphrase = 0
generate = 1
eval_ppl = 1
gpus = [0,1]
max_workers = len(gpus)
base_output_dir = "output/colm_other_models"
# steps = [0, 500, 1000, 1500, 2000, 2500]
steps = [0]
CHECKPOINT_DIR_SUFFIX = "_config2"

# Base directory for saved models
SAVED_MODELS_BASE_DIR = "saved_models_new"
CHECKPOINT_BASE_DIR = "./finetuning/colm"
# Pair each selector_matrix_dir (relative) with its corresponding model
selector_matrix_model_pairs = [
    # ("openwebtext_Olmo-3-1025-7B_k255", "allenai/Olmo-3-1025-7B"),
    ("openwebtext_SmolLM2-1.7B_k256", "HuggingFaceTB/SmolLM2-1.7B"),
    # ("openwebtext_Qwen2.5-7B_k256", "Qwen/Qwen2.5-7B"),
    # ("Dolci-Instruct-SFT_phi-4_k252", "microsoft/phi-4"),
    # ("openwebtext_Mistral-7B-v0.3_k255", "mistralai/Mistral-7B-v0.3"),
    # ("openwebtext_phi-4_k198", "microsoft/phi-4")
]

selector_matrix_model_pairs = [
    # ("openwebtext_Llama-2-7b-hf_k256", "meta-llama/Llama-2-7b-hf"),
    ("openwebtext_Llama-2-7b-hf_k254_semalign_contrastive_Qwen3-Embedding-8B",
     "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k8_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k16_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k32_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k64_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k96_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k127_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k192_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
    # ("openwebtext_Llama-2-7b-hf_k384_semalign_contrastive_Qwen3-Embedding-8B",
    #  "meta-llama/Llama-2-7b-hf"),
]

selector_matrix_model_pairs = [
    # ("openwebtext_Olmo-3-1025-7B_k253_semalign_contrastive_Qwen3-Embedding-8B",
    #  "allenai/Olmo-3-1025-7B"),
    # ("openwebtext_Qwen2.5-7B_k251_semalign_contrastive_Qwen3-Embedding-8B",
    #  "Qwen/Qwen2.5-7B"),
    ("openwebtext_SmolLM2-1.7B_k254_semalign_contrastive_Qwen3-Embedding-8B",
     "HuggingFaceTB/SmolLM2-1.7B"),
    # ("openwebtext_phi-4_k250_semalign_contrastive_Qwen3-Embedding-8B",
    #  "microsoft/phi-4"),
]

# selector_matrix_model_pairs = [
#     ("openwebtext_Mistral-7B-v0.3_k254_semalign_contrastive_Qwen3-Embedding-8B",
#      "mistralai/Mistral-7B-v0.3"),
#     ("openwebtext_Mistral-7B-v0.3_k255",
#      "mistralai/Mistral-7B-v0.3"),
# ]

# selector_matrix_model_pairs = [
#     ("openwebtext_Qwen2.5-7B_k256",
#      "Qwen/Qwen2.5-7B"),
#     ("openwebtext_Qwen2.5-7B_k251_semalign_contrastive_Qwen3-Embedding-8B",
#      "Qwen/Qwen2.5-7B"),
# ]

# ==== Watermark-specific Params ====
gamma = delta = distributions = gaussmark_configs = None

if watermark_type == "mb" or watermark_type == "mb_binom" or watermark_type == "mb_discrete":
    gamma = [0.25]
    delta = [0.8]

elif watermark_type == "noise":
    delta = [1.25]
    distributions = ["symmetric_beta", "gaussian", "uniform"]

elif watermark_type in ["kgw", "kgw_llr", "unigram"]:
    selector_matrix_model_pairs = [
        (None, "meta-llama/Llama-2-7b-hf")
        # (None, "mistralai/Mistral-7B-v0.3")
    ]
    gamma = [0.25]
    delta = [1.5]

elif watermark_type == "distilled":
    seeds = [15485863]
    selector_matrix_model_pairs = [
        (None, "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2")]

elif watermark_type == "gaussmark":
    selector_matrix_model_pairs = [
        # (None, "meta-llama/Llama-2-7b-hf")
        (None, "mistralai/Mistral-7B-v0.3"),
        # (None, "Qwen/Qwen2.5-7B")
    ]
    gaussmark_configs = [
        ("lm_head.weight", sigma)
        for sigma in [0.004, 0.005, 0.006, 0.007]
    ]

    gaussmark_configs = [
        ("model.layers.27.mlp.up_proj.weight", sigma)
        for sigma in [0.02, 0.025, 0.03, 0.035, 0.045]
    ]

    gaussmark_configs = [
        ("model.layers.20.mlp.up_proj.weight", 0.005)
    ]

    # gaussmark_configs = [
    #     ("model.layers.27.mlp.up_proj.weight", 0.04)
    # ]

elif watermark_type == "rl":
    seeds = [15485863]
    base_dir = "/pool.ssd/users/miroojin/watermarking_rl"
    rl_model_path = f"{base_dir}/c4_llama2-7b_llama2-1.1b_b4_step2500_dosample"


def build_jobs_mb():
    return [
        (gpu, {
            'gamma': g, 'delta': d,
            'dataset': dataset, 'model': model, 'seed': seed, 'step': step,
            'selector_matrix_dir': os.path.join(SAVED_MODELS_BASE_DIR, selector_matrix_dir),
            'checkpoint_dir': os.path.join(CHECKPOINT_BASE_DIR, selector_matrix_dir + CHECKPOINT_DIR_SUFFIX + f"_seed{seed}")
        })
        for gpu, (g, d, dataset, (selector_matrix_dir, model), seed, step) in enumerate(
            itertools.product(gamma, delta, datasets, selector_matrix_model_pairs, seeds, steps))
    ]


def build_jobs_noise():
    return [
        (gpu, {
            'gamma': 0, 'delta': d,
            'distribution': dist,
            'dataset': dataset, 'model': model, 'seed': seed
        })
        for gpu, (d, dist, dataset, (_, model), seed) in enumerate(
            itertools.product(delta, distributions, datasets, selector_matrix_model_pairs, seeds))
    ]


def build_jobs_kgw():
    return [
        (gpu, {
            'gamma': g, 'delta': d,
            'dataset': dataset, 'model': model, 'seed': seed
        })
        for gpu, (g, d, dataset, (_, model), seed) in enumerate(
            itertools.product(gamma, delta, datasets, selector_matrix_model_pairs, seeds))
    ]


def build_jobs_unigram():
    return [
        (gpu, {
            'gamma': g, 'delta': d,
            'dataset': dataset, 'model': model, 'seed': seed
        })
        for gpu, (g, d, dataset, (_, model), seed) in enumerate(
            itertools.product(gamma, delta, datasets, selector_matrix_model_pairs, seeds))
    ]


def build_jobs_distilled():
    jobs = []
    product_iter = itertools.product(
        datasets, selector_matrix_model_pairs, seeds, steps)
    for gpu, (dataset, (_, model), seed, step) in enumerate(product_iter):
        checkpoint_folder = f"kgw_distilled_Llama-2-7b-hf" + CHECKPOINT_DIR_SUFFIX
        job = (
            gpu,
            {
                'gamma': 0.25,
                'delta': 0,
                'dataset': dataset,
                'model': model,
                'seed': seed,
                'step': step,
                'checkpoint_dir': os.path.join(CHECKPOINT_BASE_DIR, checkpoint_folder)
            }
        )
        jobs.append(job)
    return jobs


def build_jobs_gaussmark():
    jobs = []
    product_iter = itertools.product(
        gaussmark_configs, datasets, selector_matrix_model_pairs, seeds, steps
    )
    for gpu, (gaussmark_cfg, dataset, (_, model), seed, step) in enumerate(product_iter):
        layer, sigma = gaussmark_cfg
        model_suffix = model.split("/")[-1]
        checkpoint_folder = f"gaussmark_{model_suffix}" + \
            CHECKPOINT_DIR_SUFFIX + f"_seed{seed}"
        job = (
            gpu,
            {
                'gamma': 0,
                'delta': 0,
                'dataset': dataset,
                'model': model,
                'seed': seed,
                'target_param_name': layer,
                'sigma': sigma,
                'step': step,
                'checkpoint_dir': os.path.join(CHECKPOINT_BASE_DIR, checkpoint_folder)
            }
        )
        jobs.append(job)
    return jobs


def build_jobs_rl():
    return [
        (gpu, {
            'gamma': 0, 'delta': 0,
            'dataset': dataset, 'model': model, 'seed': seed,
            'rl_model_path': rl_model_path
        })
        for gpu, (dataset, (_, model), seed) in enumerate(
            itertools.product(datasets, selector_matrix_model_pairs, seeds))
    ]

# ==== Shared Job Runner ====


def run_job_common(args_and_locks):
    params, gpu_locks = args_and_locks
    gpu, param = params

    model_suffix = param['model'].split("/")[-1]
    output_dir = f"{base_output_dir}/{model_suffix}"
    lock = gpu_locks[gpu]
    num_samples = 1000 if param['dataset'] == "combined" else NUM_SAMPLES
    selector_matrix_dir = param.get('selector_matrix_dir', '.')
    checkpoint_dir = param.get('checkpoint_dir', '.')
    cmd = [
        './scripts/test_watermarking.sh',
        '--gamma', str(param.get('gamma', 0)),
        '--delta', str(param.get('delta', 0)),
        '--seed', str(param['seed']),
        '--paraphrase', str(paraphrase),
        '--generate', str(generate),
        '--eval_ppl', str(eval_ppl),
        '--dataset', param['dataset'],
        '--distribution', param.get('distribution', 'gaussian'),
        '--watermark', watermark_type,
        '--model', param['model'],
        '--output_dir', output_dir,
        '--target_param_name', param.get('target_param_name',
                                         'lm_head.weight'),
        '--num_samples', str(num_samples),
        '--sigma', str(param.get('sigma', 0)),
        '--rl_model_path', param.get('rl_model_path', '.'),
        '--checkpoint_dir', checkpoint_dir,
        '--step', str(param.get('step', 0)),
        '--selector_matrix_dir', selector_matrix_dir
    ]

    with lock:
        try:
            env = os.environ.copy()
            if isinstance(gpu, tuple):
                gpu = ','.join(map(str, gpu))

            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            env['CUDA_CACHE_DISABLE'] = '1'

            print(f"Executing command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid
            )
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                return f"Success on GPU {gpu}: {stdout.strip()}"
            else:
                return f"Error on GPU {gpu}:\nSTDOUT:\n{stdout.strip()}\nSTDERR:\n{stderr.strip()}"
        except Exception as e:
            return f"Exception on GPU {gpu}: {e}"


# ==== Dispatcher ====

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    gpu_locks = {gpu: manager.Semaphore(1) for gpu in gpus}

    # Select builder
    if watermark_type in ["mb", "mb_binom", "mb_discrete"]:
        jobs = build_jobs_mb()
    elif watermark_type == "noise":
        jobs = build_jobs_noise()
    elif watermark_type in ["kgw", "kgw_llr"]:
        jobs = build_jobs_kgw()
    elif watermark_type == "unigram":
        jobs = build_jobs_unigram()
    elif watermark_type == "distilled":
        jobs = build_jobs_distilled()
    elif watermark_type == "gaussmark":
        jobs = build_jobs_gaussmark()
    elif watermark_type == "rl":
        jobs = build_jobs_rl()
    else:
        raise ValueError(f"Unsupported watermark_type: {watermark_type}")

    # Round-robin assign GPU and pack job
    wrapped_jobs = [((gpus[i % len(gpus)], param), gpu_locks)
                    for i, (gpu, param) in enumerate(jobs)]
    print(f"Generated {len(wrapped_jobs)} jobs for type {watermark_type}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job_common, job)
                   for job in wrapped_jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"Job failed with exception: {e}")

    print("All jobs completed.")
