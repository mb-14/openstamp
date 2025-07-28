#!/usr/bin/env python3

import subprocess
import concurrent.futures
import itertools
import os
import multiprocessing


# Configuration
seeds = [15485863, 12997009, 22983996]
# seeds = [15485863]  # For debugging, use a single seed
align = [0]
k = [235]
gamma = [0.25]
delta = [0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
delta = [1.25]
distributions = ["symmetric_beta", "gaussian", "uniform"]
# For debugging, use a single distribution
distributions = ["gaussian"]
# distributions = ["symmetric_beta"]  # For debugging, use a single distribution
models = ["meta-llama/Llama-2-7b-hf",
          "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.1-8B"]
# models = ["meta-llama/Llama-3.1-8B"]
# For debugging, use a single model
# models = ["mistralai/Mistral-7B-v0.3"]
models = ["meta-llama/Llama-2-7b-hf"]  # For debugging, use a single model
# models = ["mbakshi1094/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1.25"]
datasets = ["arxiv", "wikipedia", "booksum"]
datasets = ["realnewslike"]
gaussmark_configs = [
    ("lm_head.weight", sigma)
    for sigma in [0.004, 0.005, 0.006, 0.007]
] + [
    ("model.layers.27.mlp.up_proj.weight", sigma)
    for sigma in [0.02, 0.025, 0.03, 0.035, 0.04]
]

gaussmark_configs = [
    ("lm_head.weight", 0.005),
    ("model.layers.27.mlp.up_proj.weight", 0.025)
]  # For debugging, use a single gaussmark config

gpus = [1, 2, 3]
max_workers = len(gpus)


def run_job(params_and_locks):
    params, gpu_locks = params_and_locks
    gpu, a, k_val, g, d, dataset, model, seed, distribution, gaussmark_layer, gaussmark_sigma = params

    lock = gpu_locks[gpu]

    model_suffix = model.split("/")[-1]

    output_dir = f"output/mb/{model_suffix}"

    with lock:
        print(
            f"Running test_watermarking on GPU {gpu} with align={a}, k={k_val}, gamma={g}, delta={d}, dataset={dataset}, seed={seed}")

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env['CUDA_CACHE_DISABLE'] = '1'

        cmd = [
            './scripts/test_watermarking.sh',
            '--gamma', str(g),
            '--delta', str(d),
            '--k', str(k_val),
            '--seed', str(seed),
            '--num_samples', '500',
            '--align', str(a),
            '--paraphrase', '1',
            '--train', '0',
            '--generate', '0',
            '--eval_ppl', '0',
            '--dataset', dataset,
            '--distribution', distribution,
            '--watermark', 'gaussmark',
            '--model', model,
            '--output_dir', output_dir,
            '--target_param_name', gaussmark_layer,
            '--sigma', str(gaussmark_sigma),
        ]

        try:
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


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    gpu_locks = {gpu: manager.Semaphore(1) for gpu in gpus}

    jobs = []
    gpu_idx = 0
    for a, k_val, g, d, dataset, model, seed, distribution, (gaussmark_layer, gaussmark_weight) in itertools.product(align, k, gamma, delta, datasets, models, seeds, distributions, gaussmark_configs):
        gpu = gpus[gpu_idx % len(gpus)]
        jobs.append(
            ((gpu, a, k_val, g, d, dataset, model, seed, distribution, gaussmark_layer, gaussmark_weight), gpu_locks))
        gpu_idx += 1

    print(f"Generated {len(jobs)} jobs")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Job failed with exception: {e}")

    print("All jobs completed!")
