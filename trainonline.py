from google.cloud import aiplatform

aiplatform.init(
    project="dtumlops",
    location="europe-west2",
)

job = aiplatform.CustomTrainingJob(
    display_name="my-training-job",
    script_path="train.py",
    container_uri="gcr.io/mlops_99/custom-train-image",
)

job.run(
    args=["--data_path=gs://mlops_99/data", "--epochs=20"],
    machine_type="n1-standard-4"
)
