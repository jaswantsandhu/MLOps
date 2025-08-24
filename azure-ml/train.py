from config import ml_client
from azure.ai.ml import command
from azure.ai.ml.entities import AmlCompute


def create_compute():
    """Create compute cluster in UK South and return its name"""
    cpu_compute_target = "cpu-cluster"

    try:
        cpu_cluster = ml_client.compute.get(cpu_compute_target)
        print(f"Found existing cluster: {cpu_cluster.name}")
    except Exception:
        print("Creating new compute cluster…")
        cpu_cluster = AmlCompute(
            name=cpu_compute_target,
            type="amlcompute",
            size="STANDARD_E8S_V3",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=180,
            tier="Dedicated",
            location="uksouth",
        )
        cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster).result()
        print(f"Created cluster: {cpu_cluster.name}")

    return cpu_cluster.name


def submit_training_job(compute_name: str):
    """Submit a training job and wait for completion."""
    job = command(
        inputs={
            "test_train_ratio": 0.2,
            "n_estimators": 50,
            "max_depth": 10,
            "registered_model_name": "iris_model",
        },
        code="./src/",
        command=(
            "python main.py "
            "--test_train_ratio ${{inputs.test_train_ratio}} "
            "--n_estimators ${{inputs.n_estimators}} "
            "--max_depth ${{inputs.max_depth}} "
            "--registered_model_name ${{inputs.registered_model_name}}"
        ),
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
        compute=compute_name,
        experiment_name="iris-training-experiment",
        display_name="iris_classification_training",
    )

    job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted: {job.name}")
    print("Waiting for training job to complete…")
    ml_client.jobs.stream(job.name)

    completed = ml_client.jobs.get(job.name)
    print(f"Job status: {completed.status}")
    if completed.status != "Completed":
        raise Exception(f"Training job failed: {completed.status}")
    print("✅ Training job completed successfully")
    return job.name


if __name__ == "__main__":
    print("Step 1: Creating compute cluster...")
    compute_name = create_compute()

    print("\nStep 2: Submitting training job...")
    submit_training_job(compute_name)

    print("\nTraining complete")

