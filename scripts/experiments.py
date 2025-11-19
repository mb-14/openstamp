#!/usr/bin/env python3

import subprocess
import concurrent.futures
import itertools
import os
import multiprocessing

# ==== Common Config ====
seeds = [15485863, 12997009, 22983996]
seeds = [15485863]
models = ["meta-llama/Llama-2-7b-hf"]
# models = ["google/gemma-2-27b"]
# models = ["mistralai/Mistral-7B-v0.3"]
datasets = ["arxiv", "wikipedia", "booksum"]
datasets = ["realnewslike"]
# Options: mb, mb_binom, noise, kgw, kgw_llr, distilled, gaussmark, rl, binoc
watermark_type = "gaussmark"
paraphrase = 0
generate = 1
eval_ppl = 1
gpus = [1, 2, 3]
max_workers = len(gpus)
base_output_dir = "output/detector"
steps = [0, 500, 1000, 1500, 2000, 2500, 3000]
steps = [0]
checkpoint_dir = "."
# ==== Watermark-specific Params ====
k = gamma = delta = distributions = gaussmark_configs = None

if watermark_type == "mb" or watermark_type == "mb_binom" or watermark_type == "mb_discrete":
    checkpoint_dir = "./lora_targeted"
    k = [235]
    gamma = [0.25]
    delta = [1.0]

elif watermark_type == "noise":
    delta = [1.25]
    distributions = ["symmetric_beta", "gaussian", "uniform"]

elif watermark_type in ["kgw", "kgw_llr"]:
    gamma = [0.25]
    delta = [1.2]

elif watermark_type == "distilled":
    seeds = [15485863]
    models = ["cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"]

elif watermark_type == "gaussmark":
    gaussmark_configs = [
        ("lm_head.weight", sigma)
        for sigma in [0.004, 0.005, 0.006, 0.007]
    ] + [
        ("model.layers.27.mlp.up_proj.weight", sigma)
        for sigma in [0.02, 0.025, 0.03, 0.035, 0.04]
    ]

    gaussmark_configs = [
        ("model.layers.20.mlp.up_proj.weight", 0.005)
    ]
    
    gaussmark_configs = [
        ("model.layers.27.mlp.up_proj.weight", 0.04)
    ]
  
elif watermark_type == "rl":
    seeds = [15485863]
    base_dir = "/pool.ssd/users/miroojin/watermarking_rl"
    rl_model_path = f"{base_dir}/c4_llama2-7b_llama2-1.1b_b4_step2500_dosample"
elif watermark_type == "binoc":
    seeds = [15485863]
    base_dir = "./binoculars_ft_c4"
    binoc_performer_lora_path = f"{base_dir}/performer_model_lora_L0_2.0_lambda_0.005"
    binoc_observer_lora_path = f"{base_dir}/observer_model_lora_L0_2.0_lambda_0.005"


def build_jobs_mb():
    return [
        (gpu, {
            'k': k_val, 'gamma': g, 'delta': d,
            'dataset': dataset, 'model': model, 'seed': seed, 'step': step
        })
        for gpu, (k_val, g, d, dataset, model, seed, step) in enumerate(
            itertools.product(k, gamma, delta, datasets, models, seeds, steps))
    ]


def build_jobs_noise():
    return [
        (gpu, {
            'k': 0, 'gamma': 0, 'delta': d,
            'distribution': dist,
            'dataset': dataset, 'model': model, 'seed': seed
        })
        for gpu, (d, dist, dataset, model, seed) in enumerate(
            itertools.product(delta, distributions, datasets, models, seeds))
    ]


def build_jobs_kgw():
    return [
        (gpu, {
            'k': 0, 'gamma': g, 'delta': d,
            'dataset': dataset, 'model': model, 'seed': seed
        })
        for gpu, (g, d, dataset, model, seed) in enumerate(
            itertools.product(gamma, delta, datasets, models, seeds))
    ]


def build_jobs_distilled():
    return [
        (gpu, {
            'k': 0, 'gamma': 0.25, 'delta': 0,
            'dataset': dataset, 'model': model, 'seed': seed, 'step': step
        })
        for gpu, (dataset, model, seed, step) in enumerate(
            itertools.product(datasets, models, seeds, steps))
    ]


def build_jobs_gaussmark():
    return [
        (gpu, {
            'k': 0, 'gamma': 0, 'delta': 0,
            'dataset': dataset, 'model': model, 'seed': seed,
            'target_param_name': layer, 'sigma': sigma, 'step': step
        })
        for gpu, ((layer, sigma), dataset, model, seed, step) in enumerate(
            itertools.product(gaussmark_configs, datasets, models, seeds, steps))
    ]


def build_jobs_rl():
    return [
        (gpu, {
            'k': 0, 'gamma': 0, 'delta': 0,
            'dataset': dataset, 'model': model, 'seed': seed,
            'rl_model_path': rl_model_path
        })
        for gpu, (dataset, model, seed) in enumerate(
            itertools.product(datasets, models, seeds))
    ]


def build_jobs_binoc():
    return [
        (gpu, {
            'k': 0, 'gamma': 0, 'delta': 0,
            'dataset': dataset, 'model': model, 'seed': seed,
            'performer_lora_path': binoc_performer_lora_path,
            'observer_lora_path': binoc_observer_lora_path
        })
        for gpu, (dataset, model, seed) in enumerate(
            itertools.product(datasets, models, seeds))
    ]

# ==== Shared Job Runner ====


def run_job_common(args_and_locks):
    params, gpu_locks = args_and_locks
    gpu, param = params

    model_suffix = param['model'].split("/")[-1]
    output_dir = f"{base_output_dir}/{model_suffix}"
    lock = gpu_locks[gpu]
    num_samples = 1000 if param['dataset'] == "combined" else 500
    cmd = [
        './scripts/test_watermarking.sh',
        '--gamma', str(param.get('gamma', 0)),
        '--delta', str(param.get('delta', 0)),
        '--k', str(param.get('k', 0)),
        '--seed', str(param['seed']),
        '--num_samples', '500',
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
        '--binoc_performer_lora_path', param.get('performer_lora_path', '.'),
        '--binoc_observer_lora_path', param.get('observer_lora_path', '.'),
        '--checkpoint_dir', checkpoint_dir,
        '--step', str(param.get('step', 0))
    ]

    with lock:
        try:
            env = os.environ.copy()
            if isinstance(gpu, tuple):
                gpu = ','.join(map(str, gpu))

            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            env['CUDA_CACHE_DISABLE'] = '1'

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
    elif watermark_type == "distilled":
        jobs = build_jobs_distilled()
    elif watermark_type == "gaussmark":
        jobs = build_jobs_gaussmark()
    elif watermark_type == "rl":
        jobs = build_jobs_rl()
    elif watermark_type == "binoc":
        jobs = build_jobs_binoc()
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
