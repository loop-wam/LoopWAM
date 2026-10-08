import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from fastwam.datasets.lerobot.robot_video_dataset import (
    build_robotwin_prompt,
    normalize_prompt_quality_score_list,
    normalize_prompt_quality_suffix,
    normalize_prompt_quality_suffix_list,
    normalize_text_embedding_cache_dirs,
)
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _quality_score_cache_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _parse_episode_success(record: dict[str, Any]) -> bool:
    def parse_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y", "success", "succeeded"}:
                return True
            if text in {"false", "0", "no", "n", "failure", "failed"}:
                return False
        return None

    for key in ("success", "eval_success", "episode_success"):
        parsed = parse_bool(record.get(key))
        if parsed is not None:
            return parsed

    raw_file_name = str(record.get("raw_file_name", ""))
    match = re.search(r"success[-_=](true|false|1|0)", raw_file_name, flags=re.IGNORECASE)
    if match is not None:
        parsed = parse_bool(match.group(1))
        if parsed is not None:
            return parsed

    raise ValueError(
        "Unable to infer episode success from meta/episodes.jsonl. "
        "Expected `success` field or `success-true/false` in `raw_file_name`."
    )


def _resolve_episode_quality_score(score_spec: Any, episode_record: dict[str, Any]) -> int | None:
    if score_spec is None:
        return None
    if isinstance(score_spec, dict):
        success = _parse_episode_success(episode_record)
        return int(score_spec["success"] if success else score_spec["failure"])
    return int(score_spec)


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig):
    prompt_sources: list[tuple[str, str | None, Any, Path]] = []
    seen_sources: set[tuple[str, str | None, str | None, Path]] = set()
    cache_dirs: list[Path] = []
    context_lens = set()

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None:
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )

        raw_dirs_list = [str(ds) for ds in raw_dirs]
        cache_dir_list = normalize_text_embedding_cache_dirs(
            cache_dir,
            len(raw_dirs_list),
            field_name=f"{node_path}.text_embedding_cache_dir",
        )
        if cache_dir_list is None:
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )
        quality_suffixes = normalize_prompt_quality_suffix_list(
            node.get("dataset_prompt_quality_suffix"),
            len(raw_dirs_list),
            field_name=f"{node_path}.dataset_prompt_quality_suffix",
        )
        quality_scores = normalize_prompt_quality_score_list(
            node.get("dataset_prompt_quality_score"),
            len(raw_dirs_list),
            field_name=f"{node_path}.dataset_prompt_quality_score",
        )
        for ds_str, quality_suffix, quality_score, cache_dir_str in zip(
            raw_dirs_list,
            quality_suffixes,
            quality_scores,
            cache_dir_list,
        ):
            cache_dir_path = Path(cache_dir_str).expanduser()
            if cache_dir_path not in cache_dirs:
                cache_dirs.append(cache_dir_path)
            source_key = (ds_str, quality_suffix, _quality_score_cache_key(quality_score), cache_dir_path)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                prompt_sources.append((ds_str, quality_suffix, quality_score, cache_dir_path))

        context_len = node.get("context_len")
        if context_len is not None:
            context_lens.add(int(context_len))

        logger.info(
            "Discovered dataset node `%s` with %d dataset_dirs (quality suffixes: %s, quality scores: %s).",
            node_path,
            len(raw_dirs_list),
            quality_suffixes,
            quality_scores,
        )

    return prompt_sources, cache_dirs, context_lens


def _resolve_context_len(context_lens: set[int]) -> int:
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _read_unique_prompts(prompt_sources: list[tuple[str, str | None, Any, Path]]) -> list[tuple[str, list[Path]]]:
    prompt_targets: list[tuple[str, list[Path]]] = []
    prompt_to_cache_dirs: dict[str, list[Path]] = {}
    total_task_rows = 0

    for ds_dir, quality_suffix, quality_score, cache_dir in prompt_sources:
        ds_path = Path(ds_dir)
        if isinstance(quality_score, dict):
            episodes_path = ds_path / "meta" / "episodes.jsonl"
            if not episodes_path.exists():
                raise FileNotFoundError(f"Missing episodes file: {episodes_path}")

            with episodes_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    tasks = record.get("tasks")
                    if not tasks:
                        raise KeyError(f"Missing `tasks` field at {episodes_path}:{line_idx}")
                    resolved_score = _resolve_episode_quality_score(quality_score, record)
                    for task_raw in tasks:
                        task = str(task_raw).strip()
                        if not task:
                            continue
                        prompt = build_robotwin_prompt(task, quality_suffix, quality_score=resolved_score)
                        total_task_rows += 1
                        if prompt not in prompt_to_cache_dirs:
                            prompt_to_cache_dirs[prompt] = []
                            prompt_targets.append((prompt, prompt_to_cache_dirs[prompt]))
                        if cache_dir not in prompt_to_cache_dirs[prompt]:
                            prompt_to_cache_dirs[prompt].append(cache_dir)
            continue

        tasks_path = ds_path / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")

        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                task = str(record["task"])
                prompt = build_robotwin_prompt(task, quality_suffix, quality_score=quality_score)
                total_task_rows += 1
                if prompt not in prompt_to_cache_dirs:
                    prompt_to_cache_dirs[prompt] = []
                    prompt_targets.append((prompt, prompt_to_cache_dirs[prompt]))
                if cache_dir not in prompt_to_cache_dirs[prompt]:
                    prompt_to_cache_dirs[prompt].append(cache_dir)

    logger.info(
        "Loaded %d task rows from %d dataset sources, deduplicated to %d prompts.",
        total_task_rows,
        len(prompt_sources),
        len(prompt_targets),
    )
    return prompt_targets


def _get_override_prompt(
    override_instruction: Any,
    override_quality_suffix: Any = None,
) -> str | None:
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    return build_robotwin_prompt(task, normalize_prompt_quality_suffix(override_quality_suffix))


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute_text_embeds.py",
            torch.cuda.device_count(),
        )

    overwrite = _to_bool(cfg.get("overwrite", True))
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
        logger.info("Using override_instruction; skipping dataset scan and encoding exactly 1 prompt.")
    else:
        if not prompt_sources:
            raise ValueError("No `dataset_dirs` found under `cfg.data`.")
        prompt_targets = _read_unique_prompts(prompt_sources)
    if not prompt_targets:
        logger.warning("No prompts found from tasks.jsonl; nothing to do.")
        return

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    enc_id = _model_id_to_enc_id(model_id)

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    stats = {
        str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
        for cache_dir in cache_dirs
    }

    prompt_targets = prompt_targets[rank::world_size] if is_distributed else prompt_targets

    if not overwrite:
        fully_cached_local = 0
        targets_to_encode: list[tuple[str, list[Path]]] = []
        for prompt, target_cache_dirs in prompt_targets:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            fully_cached = True
            for cache_dir in target_cache_dirs:
                cache_path = cache_dir / filename
                if not cache_path.exists():
                    fully_cached = False
                    break
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in target_cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                targets_to_encode.append((prompt, target_cache_dirs))

        prompt_targets = targets_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompt_targets)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompt_targets)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompt_targets)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    over_length_prompts = 0
    with tqdm(
        total=len(prompt_targets),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompt_targets), DEFAULT_BATCH_SIZE):
                batch_targets = prompt_targets[start : start + DEFAULT_BATCH_SIZE]
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

                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [
                [stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
