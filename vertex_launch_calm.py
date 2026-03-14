#!/usr/bin/env python3
import argparse
from datetime import datetime
from textwrap import dedent

from google.cloud import aiplatform


DOWNLOAD_HELPER = dedent(
    r"""
    python - <<'PY'
    import os
    from pathlib import Path
    from google.cloud import storage

    client = storage.Client()

    def parse_gs(uri: str):
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got {uri}")
        rest = uri[5:]
        bucket, _, blob = rest.partition("/")
        return bucket, blob

    def download_file(uri: str, dst: str):
        bucket, blob = parse_gs(uri)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        client.bucket(bucket).blob(blob).download_to_filename(dst)

    def download_prefix(uri: str, dst_dir: str):
        bucket, prefix = parse_gs(uri)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix.rstrip("/")
        for blob in client.list_blobs(bucket, prefix=prefix + "/"):
            rel = blob.name[len(prefix) + 1 :]
            if not rel:
                continue
            out = dst_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(out))

    download_file(os.environ["TRAIN_DATA_GCS"], "/tmp/data/train.json")
    download_file(os.environ["VALID_DATA_GCS"], "/tmp/data/valid.json")

    ae_ckpt = os.environ.get("AE_CKPT_GCS")
    if ae_ckpt:
        download_prefix(ae_ckpt, "/tmp/ae_ckpt")
    PY
    """
).strip()


UPLOAD_HELPER = dedent(
    r"""
    python - <<'PY'
    import os
    from pathlib import Path
    from google.cloud import storage

    client = storage.Client()

    def parse_gs(uri: str):
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got {uri}")
        rest = uri[5:]
        bucket, _, blob = rest.partition("/")
        return bucket, blob.rstrip("/")

    src_dir = Path(os.environ["LOCAL_OUTPUT_DIR"])
    bucket_name, prefix = parse_gs(os.environ["GCS_OUTPUT_PREFIX"])

    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(path))
    PY
    """
).strip()


def make_worker_pool(args: argparse.Namespace, command: str, env: dict[str, str]):
    return [{
        "machine_spec": {
            "machine_type": args.machine_type,
            "accelerator_type": args.accelerator_type,
            "accelerator_count": args.accelerator_count,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": args.image_uri,
            "command": ["bash", "-lc"],
            "args": [command],
            "env": [{"name": k, "value": v} for k, v in env.items()],
        },
    }]


def run_job(args: argparse.Namespace, display_name: str, command: str, env: dict[str, str]):
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=make_worker_pool(args, command, env),
        base_output_dir=f"{args.bucket}/vertex-output/{args.run_name}/{display_name}",
    )
    run_kwargs = {"sync": True}
    if args.service_account:
        run_kwargs["service_account"] = args.service_account
    job.run(**run_kwargs)
    return job


def main():
    parser = argparse.ArgumentParser(description="Launch AE + Energy + JEPA training jobs on Vertex AI.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True, help="Base gs:// bucket/prefix, e.g. gs://my-bucket")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--train-data-gcs", required=True, help="gs://... training json/jsonl file")
    parser.add_argument("--valid-data-gcs", required=True, help="gs://... validation json/jsonl file")
    parser.add_argument("--run-name", default=f"calm-jepa-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--machine-type", default="a2-highgpu-1g")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--service-account", default=None)
    args = parser.parse_args()

    aiplatform.init(
        project=args.project_id,
        location=args.region,
        staging_bucket=f"{args.bucket}/staging",
    )

    run_prefix = f"{args.bucket}/runs/{args.run_name}"
    common_env = {
        "TRAIN_DATA_GCS": args.train_data_gcs,
        "VALID_DATA_GCS": args.valid_data_gcs,
    }

    ae_out = f"{run_prefix}/ae_micro"
    energy_out = f"{run_prefix}/energy_micro"
    jepa_out = f"{run_prefix}/jepa_micro"

    ae_command = dedent(
        f"""
        set -euo pipefail
        export PYTHONPATH=/app/CS224n-Project:/app/CS224n-Project/language-modelling
        {DOWNLOAD_HELPER}
        cd /app/CS224n-Project/language-modelling
        python -m train.train_autoencoder \\
          --tokenizer_name /app/CS224n-Project/language-modelling/llama3_tokenizer \\
          --train_file /tmp/data/train.json \\
          --validation_file /tmp/data/valid.json \\
          --config_overrides "latent_size=64,num_encoder_layers=2,num_decoder_layers=2,patch_size=4,hidden_size=256,intermediate_size=768" \\
          --keep_linebreaks True \\
          --weight_decay 0.1 \\
          --warmup_steps 100 \\
          --block_size 512 \\
          --adam_beta1 0.9 \\
          --adam_beta2 0.95 \\
          --max_grad_norm 1.0 \\
          --seed 1 \\
          --per_device_train_batch_size 8 \\
          --per_device_eval_batch_size 4 \\
          --gradient_accumulation_steps 1 \\
          --max_steps 2000 \\
          --save_strategy steps \\
          --save_steps 1000 \\
          --evaluation_strategy steps \\
          --eval_steps 500 \\
          --learning_rate 3e-4 \\
          --lr_scheduler_type constant \\
          --logging_steps 50 \\
          --do_train \\
          --do_eval \\
          --save_safetensors False \\
          --output_dir /tmp/outputs/ae_micro \\
          --overwrite_output_dir \\
          --bf16 True
        export LOCAL_OUTPUT_DIR=/tmp/outputs/ae_micro
        export GCS_OUTPUT_PREFIX={ae_out}
        {UPLOAD_HELPER}
        """
    ).strip()

    ae_job = run_job(args, f"{args.run_name}-ae", ae_command, common_env)

    post_ae_env = {
        **common_env,
        "AE_CKPT_GCS": ae_out,
    }

    energy_command = dedent(
        f"""
        set -euo pipefail
        export PYTHONPATH=/app/CS224n-Project:/app/CS224n-Project/language-modelling
        {DOWNLOAD_HELPER}
        cd /app/CS224n-Project/language-modelling
        python -m train.train_calm \\
          --ae_name_or_path /tmp/ae_ckpt \\
          --tokenizer_name /app/CS224n-Project/language-modelling/llama3_tokenizer \\
          --train_file /tmp/data/train.json \\
          --validation_file /tmp/data/valid.json \\
          --config_overrides "model_type=energy,patch_size=4,latent_size=64,hidden_size=256,intermediate_size=768,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,num_mlp_layers=2,num_samples=4,noise_size=64,beta=1.0" \\
          --keep_linebreaks True \\
          --weight_decay 0.1 \\
          --warmup_steps 100 \\
          --block_size 1024 \\
          --adam_beta1 0.9 \\
          --adam_beta2 0.95 \\
          --max_grad_norm 1.0 \\
          --seed 1 \\
          --per_device_train_batch_size 8 \\
          --per_device_eval_batch_size 4 \\
          --gradient_accumulation_steps 1 \\
          --max_steps 2000 \\
          --save_strategy steps \\
          --save_steps 1000 \\
          --evaluation_strategy steps \\
          --eval_steps 500 \\
          --learning_rate 3e-4 \\
          --lr_scheduler_type constant \\
          --logging_steps 50 \\
          --do_train \\
          --do_eval \\
          --save_safetensors False \\
          --output_dir /tmp/outputs/energy_micro \\
          --overwrite_output_dir \\
          --bf16 True
        export LOCAL_OUTPUT_DIR=/tmp/outputs/energy_micro
        export GCS_OUTPUT_PREFIX={energy_out}
        {UPLOAD_HELPER}
        """
    ).strip()

    energy_job = run_job(args, f"{args.run_name}-energy", energy_command, post_ae_env)

    jepa_command = dedent(
        f"""
        set -euo pipefail
        export PYTHONPATH=/app/CS224n-Project:/app/CS224n-Project/language-modelling
        {DOWNLOAD_HELPER}
        cd /app/CS224n-Project/language-modelling
        python -m train.train_calm \\
          --ae_name_or_path /tmp/ae_ckpt \\
          --tokenizer_name /app/CS224n-Project/language-modelling/llama3_tokenizer \\
          --train_file /tmp/data/train.json \\
          --validation_file /tmp/data/valid.json \\
          --config_overrides "model_type=jepa,patch_size=4,latent_size=64,hidden_size=256,intermediate_size=768,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,num_mlp_layers=2,lambda_sigreg=0.1,sigreg_num_slices=128,jepa_head_type=mlp,jepa_sample_noise_std=0.05,jepa_eval_noise_std=0.02" \\
          --keep_linebreaks True \\
          --weight_decay 0.1 \\
          --warmup_steps 100 \\
          --block_size 1024 \\
          --adam_beta1 0.9 \\
          --adam_beta2 0.95 \\
          --max_grad_norm 1.0 \\
          --seed 1 \\
          --per_device_train_batch_size 8 \\
          --per_device_eval_batch_size 4 \\
          --gradient_accumulation_steps 1 \\
          --max_steps 3000 \\
          --save_strategy steps \\
          --save_steps 1000 \\
          --evaluation_strategy steps \\
          --eval_steps 500 \\
          --learning_rate 3e-4 \\
          --lr_scheduler_type constant \\
          --logging_steps 50 \\
          --do_train \\
          --do_eval \\
          --save_safetensors False \\
          --output_dir /tmp/outputs/jepa_micro \\
          --overwrite_output_dir \\
          --bf16 True
        export LOCAL_OUTPUT_DIR=/tmp/outputs/jepa_micro
        export GCS_OUTPUT_PREFIX={jepa_out}
        {UPLOAD_HELPER}
        """
    ).strip()

    jepa_job = run_job(args, f"{args.run_name}-jepa", jepa_command, post_ae_env)

    print("Completed jobs:")
    print("AE:", ae_job.resource_name)
    print("Energy:", energy_job.resource_name)
    print("JEPA:", jepa_job.resource_name)
    print("Run outputs:", run_prefix)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
from datetime import datetime
from textwrap import dedent

from google.cloud import aiplatform


DOWNLOAD_HELPER = dedent(
    r"""
    python - <<'PY'
    import os
    from pathlib import Path
    from google.cloud import storage

    client = storage.Client()

    def parse_gs(uri: str):
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got {uri}")
        rest = uri[5:]
        bucket, _, blob = rest.partition("/")
        return bucket, blob

    def download_file(uri: str, dst: str):
        bucket, blob = parse_gs(uri)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        client.bucket(bucket).blob(blob).download_to_filename(dst)

    def download_prefix(uri: str, dst_dir: str):
        bucket, prefix = parse_gs(uri)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for blob in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"):
            rel = blob.name[len(prefix.rstrip("/") + "/"):]
            if not rel:
                continue
            out = dst_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(out))

    download_file(os.environ["TRAIN_DATA_GCS"], "/tmp/data/train.json")
    download_file(os.environ["VALID_DATA_GCS"], "/tmp/data/valid.json")

    ae_ckpt = os.environ.get("AE_CKPT_GCS")
    if ae_ckpt:
        download_prefix(ae_ckpt, "/tmp/ae_ckpt")
    PY
    """
).strip()


UPLOAD_HELPER = dedent(
    r"""
    python - <<'PY'
    import os
    from pathlib import Path
    from google.cloud import storage

    client = storage.Client()

    def parse_gs(uri: str):
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got {uri}")
        rest = uri[5:]
        bucket, _, blob = rest.partition("/")
        return bucket, blob.rstrip("/")

    src_dir = Path(os.environ["LOCAL_OUTPUT_DIR"])
    bucket_name, prefix = parse_gs(os.environ["GCS_OUTPUT_PREFIX"])

    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(path))
    PY
    """
).strip()


def make_worker_pool(args: argparse.Namespace, command: str, env: dict[str, str]):
    return [{
        "machine_spec": {
            "machine_type": args.machine_type,
            "accelerator_type": args.accelerator_type,
            "accelerator_count": args.accelerator_count,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": args.image_uri,
            "command": ["bash", "-lc"],
            "args": [command],
            "env": [{"name": k, "value": v} for k, v in env.items()],
        },
    }]


def run_job(args: argparse.Namespace, display_name: str, command: str, env: dict[str, str]):
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=make_worker_pool(args, command, env),
        base_output_dir=f"{args.bucket}/vertex-output/{args.run_name}/{display_name}",
    )
    run_kwargs = {"sync": True}
    if args.service_account:
        run_kwargs["service_account"] = args.service_account
    job.run(**run_kwargs)
    return job


def main():
    parser = argparse.ArgumentParser(description="Launch AE + Energy + JEPA custom jobs on Vertex AI.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True, help="Base gs:// bucket/prefix, e.g. gs://my-bucket")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--train-data-gcs", required=True, help="gs://... training json/jsonl file")
    parser.add_argument("--valid-data-gcs", required=True, help="gs://... validation json/jsonl file")
    parser.add_argument("--run-name", default=f"calm-jepa-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--machine-type", default="a2-highgpu-1g")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--service-account", default=None)
    args = parser.parse_args()

    aiplatform.init(
        project=args.project_id,
        location=args.region,
        staging_bucket=f"{args.bucket}/staging",
    )

    run_prefix = f"{args.bucket}/runs/{args.run_name}"
    common_env = {
        "TRAIN_DATA_GCS": args.train_data_gcs,
        "VALID_DATA_GCS": args.valid_data_gcs,
    }

    ae_out = f"{run_prefix}/ae_micro"
    energy_out = f"{run_prefix}/energy_micro"
    jepa_out = f"{run_prefix}/jepa_micro"

    ae_command = dedent(
        f"""
        set -euo pipefail
        export PYTHONPATH=/app/CS224n-Project:/app/CS224n-Project/language-modelling
        {DOWNLOAD_HELPER}
        cd /app/CS224n-Project/language-modelling
        python -m train.train_autoencoder \\
          --tokenizer_name /app/CS224n-Project/language-modelling/llama3_tokenizer \\
          --train_file /tmp/data/train.json \\
          --validation_file /tmp/data/valid.json \\
          --config_overrides "latent_size=64,num_encoder_layers=2,num_decoder_layers=2,patch_size=4,hidden_size=256,intermediate_size=768" \\
          --keep_linebreaks True \\
          --weight_decay 0.1 \\
          --warmup_steps 100 \\
          --block_size 512 \\
          --adam_beta1 0.9 \\
          --adam_beta2 0.95 \\
          --max_grad_norm 1.0 \\
          --seed 1 \\
          --per_device_train_batch_size 8 \\
          --per_device_eval_batch_size 4 \\
          --gradient_accumulation_steps 1 \\
          --max_steps 2000 \\
          --save_strategy steps \\
          --save_steps 1000 \\
          --evaluation_strategy steps \\
          --eval_steps 500 \\
          --learning_rate 3e-4 \\
          --lr_scheduler_type constant \\
          --logging_steps 50 \\
          --do_train \\
          --do_eval \\
          --save_safetensors False \\
          --output_dir /tmp/outputs/ae_micro \\
          --overwrite_output_dir \\
          --bf16 True
        export LOCAL_OUTPUT_DIR=/tmp/outputs/ae_micro
        export GCS_OUTPUT_PREFIX={ae_out}
        {UPLOAD_HELPER}
        """
    ).strip()

    ae_job = run_job(args, f"{args.run_name}-ae", ae_command, common_env)

    post_ae_env = {
        **common_env,
        "AE_CKPT_GCS": ae_out,
    }

    energy_command = dedent(
        f"""
        set -euo pipefail
        export PYTHONPATH=/app/CS224n-Project:/app/CS224n-Project/language-modelling
        {DOWNLOAD_HELPER}
        cd /app/CS224n-Project/language-modelling
        python -m train.train_calm \\
          --ae_name_or_path /tmp/ae_ckpt \\
          --tokenizer_name /app/CS224n-Project/language-modelling/llama3_tokenizer \\
          --train_file /tmp/data/train.json \\
          --validation_file /tmp/data/valid.json \\
          --config_overrides "model_type=energy,patch_size=4,latent_size=64,hidden_size=256,intermediate_size=768,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,num_mlp_layers=2,num_samples=4,noise_size=64,beta=1.0" \\
          --keep_linebreaks True \\
          --weight_decay 0.1 \\
          --warmup_steps 100 \\
          --block_size 1024 \\
          --adam_beta1 0.9 \\
          --adam_beta2 0.95 \\
          --max_grad_norm 1.0 \\
          --seed 1 \\
          --per_device_train_batch_size 8 \\
          --per_device_eval_batch_size 4 \\
          --gradient_accumulation_steps 1 \\
          --max_steps 2000 \\
          --save_strategy steps \\
          --save_steps 1000 \\
          --evaluation_strategy steps \\
          --eval_steps 500 \\
          --learning_rate 3e-4 \\
          --lr_scheduler_type constant \\
          --logging_steps 50 \\
          --do_train \\
          --do_eval \\
          --save_safetensors False \\
          --output_dir /tmp/outputs/energy_micro \\
          --overwrite_output_dir \\
          --bf16 True
        export LOCAL_OUTPUT_DIR=/tmp/outputs/energy_micro
        export GCS_OUTPUT_PREFIX={energy_out}
        {UPLOAD_HELPER}
        """
    ).strip()

    energy_job = run_job(args, f"{args.run_name}-energy", energy_command, post_ae_env)

    jepa_command = dedent(
        f"""
        set -euo pipefail
        export PYTHONPATH=/app/CS224n-Project:/app/CS224n-Project/language-modelling
        {DOWNLOAD_HELPER}
        cd /app/CS224n-Project/language-modelling
        python -m train.train_calm \\
          --ae_name_or_path /tmp/ae_ckpt \\
          --tokenizer_name /app/CS224n-Project/language-modelling/llama3_tokenizer \\
          --train_file /tmp/data/train.json \\
          --validation_file /tmp/data/valid.json \\
          --config_overrides "model_type=jepa,patch_size=4,latent_size=64,hidden_size=256,intermediate_size=768,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=4,num_mlp_layers=2,lambda_sigreg=0.1,sigreg_num_slices=128,jepa_head_type=mlp,jepa_sample_noise_std=0.05,jepa_eval_noise_std=0.02" \\
          --keep_linebreaks True \\
          --weight_decay 0.1 \\
          --warmup_steps 100 \\
          --block_size 1024 \\
          --adam_beta1 0.9 \\
          --adam_beta2 0.95 \\
          --max_grad_norm 1.0 \\
          --seed 1 \\
          --per_device_train_batch_size 8 \\
          --per_device_eval_batch_size 4 \\
          --gradient_accumulation_steps 1 \\
          --max_steps 3000 \\
          --save_strategy steps \\
          --save_steps 1000 \\
          --evaluation_strategy steps \\
          --eval_steps 500 \\
          --learning_rate 3e-4 \\
          --lr_scheduler_type constant \\
          --logging_steps 50 \\
          --do_train \\
          --do_eval \\
          --save_safetensors False \\
          --output_dir /tmp/outputs/jepa_micro \\
          --overwrite_output_dir \\
          --bf16 True
        export LOCAL_OUTPUT_DIR=/tmp/outputs/jepa_micro
        export GCS_OUTPUT_PREFIX={jepa_out}
        {UPLOAD_HELPER}
        """
    ).strip()

    jepa_job = run_job(args, f"{args.run_name}-jepa", jepa_command, post_ae_env)

    print("Completed jobs:")
    print("AE:", ae_job.resource_name)
    print("Energy:", energy_job.resource_name)
    print("JEPA:", jepa_job.resource_name)
    print("Run outputs:", run_prefix)


if __name__ == "__main__":
    main()