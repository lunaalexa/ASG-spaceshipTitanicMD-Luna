"""
STEP 5 : Pipeline Runner
Orchestrates: ingest → preprocess → train → evaluate
"""

from data_ingestion import ingest_data
from pre_processing import preprocess
from train import train
from evaluation import evaluate

ACCURACY_THRESHOLD = 0.78


def run_pipeline():
    print("=" * 50)
    print("Step 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Preprocessing")
    train_scaled, test_scaled = preprocess()

    print("\nStep 3: Training")
    run_id = train(train_scaled, test_scaled)

    print("\nStep 4: Evaluation")
    accuracy, precision, recall = evaluate(test_scaled, run_id)

    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED for deployment (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < threshold={ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    run_pipeline()
