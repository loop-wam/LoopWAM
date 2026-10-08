import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from scripts.precompute_text_embeds import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_MODEL_ID,
    _atomic_torch_save,
    _collect_dataset_settings,
    _get_override_prompt,
    _model_id_to_enc_id,
    _read_unique_prompts,
    _resolve_context_len,
    _to_bool,
)

register_default_resolvers()
logger = get_logger(__name__)


def _setup_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = local_rank % device_count
        torch.cuda.set_device(device_index)
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def _ensure_model_files(cfg: DictConfig):
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    logger.info("Checking/downloading text encoder and tokenizer files before spawning workers.")
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    logger.info("Finished model file checks.")


def _save_payload(payload: dict[str, torch.Tensor], cache_path: Path, atomic_write: bool):
    if atomic_write:
        _atomic_torch_save(payload, cache_path)
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(cache_path))


def _worker(worker_rank: int, world_size: int, cfg_payload: dict[str, Any]):
    setup_logging(log_level=logging.INFO)
    cfg = OmegaConf.create(cfg_payload)

    rank = int(os.environ.get("RANK", worker_rank))
    local_rank = int(os.environ.get("LOCAL_RANK", worker_rank))
    device = _setup_device(local_rank=local_rank)
    logger.info("Rank %d/%d initialized on %s.", rank, world_size, device)

    try:
        overwrite = _to_bool(cfg.get("overwrite", True))
        atomic_write = _to_bool(cfg.get("atomic_write", True))
        model_cfg = cfg.model
        if model_cfg is None:
            raise ValueError("`cfg.model` is required.")
        if cfg.data is None:
            raise ValueError("`cfg.data` is required.")

        prompt_sources, cache_dirs, context_lens = _collect_dataset_settings(cfg.data)
        if not cache_dirs:
            raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")

        context_len = _resolve_context_len(context_lens)
        override_prompt = _get_override_prompt(
            cfg.get("override_instruction"),
            cfg.get("override_prompt_quality_suffix"),
        )
        if override_prompt is not None:
            prompt_targets = [(override_prompt, cache_dirs)]
            if rank == 0:
                logger.info("Using override_instruction; encoding exactly 1 prompt.")
        else:
            if not prompt_sources:
                raise ValueError("No `dataset_dirs` found under `cfg.data`.")
            prompt_targets = _read_unique_prompts(prompt_sources)
        if not prompt_targets:
            if rank == 0:
                logger.warning("No prompts found from tasks.jsonl; nothing to do.")
            return

        torch_dtype = torch.bfloat16
        batch_size = int(cfg.get("text_embed_batch_size", DEFAULT_BATCH_SIZE))
        model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
        tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
        redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
        enc_id = _model_id_to_enc_id(model_id)

        if rank == 0:
            logger.info(
                "Multi-process text embedding precompute: world_size=%d model_id=%s tokenizer_model_id=%s dtype=%s context_len=%d batch_size=%d overwrite=%s atomic_write=%s",
                world_size,
                model_id,
                tokenizer_model_id,
                torch_dtype,
                context_len,
                batch_size,
                overwrite,
                atomic_write,
            )

        _, text_config, _, tokenizer_config = _resolve_configs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
        )
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        logger.info("Rank %d resolved text encoder path: %s", rank, text_config.path)
        logger.info("Rank %d resolved tokenizer path: %s", rank, tokenizer_config.path)

        logger.info("Rank %d loading text encoder on %s.", rank, device)
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=str(device),
        ).eval()
        logger.info("Rank %d loaded text encoder.", rank)
        logger.info("Rank %d loading tokenizer.", rank)
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=context_len,
            clean="whitespace",
        )
        logger.info("Rank %d loaded tokenizer.", rank)

        stats = {
            str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
            for cache_dir in cache_dirs
        }
        prompt_targets = prompt_targets[rank::world_size]

        if not overwrite:
            fully_cached_local = 0
            targets_to_encode: list[tuple[str, list[Path]]] = []
            for prompt, target_cache_dirs in prompt_targets:
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                fully_cached = all((cache_dir / filename).exists() for cache_dir in target_cache_dirs)
                if fully_cached:
                    fully_cached_local += 1
                    for cache_dir in target_cache_dirs:
                        stats[str(cache_dir)]["skip"] += 1
                else:
                    targets_to_encode.append((prompt, target_cache_dirs))

            prompt_targets = targets_to_encode
            fully_cached_global = fully_cached_local
            to_encode_global = len(prompt_targets)
            if rank == 0:
                logger.info(
                    "overwrite=false: rank-local fully cached prompts=%d, rank-local prompts to encode=%d",
                    fully_cached_global,
                    to_encode_global,
                )

        prompts_encoded_local = len(prompt_targets)
        if rank == 0:
            logger.info("Writing caches to %d directories.", len(cache_dirs))

        over_length_prompts = 0
        with tqdm(
            total=len(prompt_targets),
            desc=f"Encoding prompts (rank {rank}/{world_size})",
            unit="prompt",
            dynamic_ncols=True,
            disable=rank != 0,
        ) as pbar:
            with torch.no_grad():
                for start in range(0, len(prompt_targets), batch_size):
                    batch_targets = prompt_targets[start : start + batch_size]
                    batch_prompts = [prompt for prompt, _ in batch_targets]
                    ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                    ids = ids.to(device)
                    mask = mask.to(device=device, dtype=torch.bool)
                    over_length_prompts += int(mask.all(dim=1).sum().item())
                    context = text_encoder(ids, mask)

                    for i, (prompt, target_cache_dirs) in enumerate(batch_targets):
                        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                        mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                        payload = {
                            "context": context_i,
                            "mask": mask_i,
                        }

                        for cache_dir in target_cache_dirs:
                            cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                            key = str(cache_dir)
                            if cache_path.exists() and not overwrite:
                                stats[key]["skip"] += 1
                                continue

                            if cache_path.exists():
                                stats[key]["overwrite"] += 1
                            else:
                                stats[key]["new"] += 1

                            _save_payload(payload, cache_path, atomic_write=atomic_write)

                    pbar.update(len(batch_prompts))

        logger.info("Rank %d finished multi-process precomputing text embeddings.", rank)
        if rank == 0:
            logger.info(
                "Rank 0 over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
                context_len,
                over_length_prompts,
                prompts_encoded_local,
            )
            for cache_dir in cache_dirs:
                key = str(cache_dir)
                logger.info(
                    "Rank 0 cache dir: %s | new=%d overwrite=%d skip=%d",
                    key,
                    stats[key]["new"],
                    stats[key]["overwrite"],
                    stats[key]["skip"],
                )
    except Exception:
        logger.exception("Rank %d failed.", rank)
        raise


def _resolve_world_size(cfg: DictConfig) -> int:
    requested = cfg.get("nproc_per_node")
    if requested is not None:
        world_size = int(requested)
        if world_size < 1:
            raise ValueError("`nproc_per_node` must be >= 1.")
        return world_size
    if torch.cuda.is_available():
        processes_per_gpu = int(cfg.get("processes_per_gpu", cfg.get("nproc_per_gpu", 1)))
        if processes_per_gpu < 1:
            raise ValueError("`processes_per_gpu` must be >= 1.")
        return max(1, torch.cuda.device_count() * processes_per_gpu)
    return 1


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1:
        _ensure_model_files(cfg)
        cfg_payload = OmegaConf.to_container(cfg, resolve=True)
        _worker(
            worker_rank=int(os.environ.get("RANK", "0")),
            world_size=env_world_size,
            cfg_payload=cfg_payload,
        )
        return

    world_size = _resolve_world_size(cfg)
    if world_size <= 1:
        _ensure_model_files(cfg)
        cfg_payload = OmegaConf.to_container(cfg, resolve=True)
        _worker(worker_rank=0, world_size=1, cfg_payload=cfg_payload)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Requested multi-process precompute, but CUDA is not available.")

    _ensure_model_files(cfg)
    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    device_count = torch.cuda.device_count()

    logger.info(
        "Spawning %d text-embedding workers across %d CUDA devices.",
        world_size,
        device_count,
    )
    mp.spawn(
        _spawn_entry,
        args=(world_size, device_count, cfg_payload),
        nprocs=world_size,
        join=True,
    )


def _spawn_entry(worker_rank: int, world_size: int, device_count: int, cfg_payload: dict[str, Any]):
    local_rank = worker_rank % device_count
    os.environ["RANK"] = str(worker_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    _worker(worker_rank=worker_rank, world_size=world_size, cfg_payload=cfg_payload)


if __name__ == "__main__":
    main()
