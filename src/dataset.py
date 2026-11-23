
from datasets import load_dataset

def load_data(dataset_name, max_examples=200):

    if not isinstance(max_examples, int) or max_examples <= 0:
        raise ValueError("max_examples must be a positive integer")

    dataset = load_dataset(
        dataset_name,
        split=f"validation[:{max_examples}]"
    )

    required_columns = ["sentence", "label"]

    for column in required_columns:
        if column not in dataset.column_names:
            raise ValueError(
                f"Missing required column: {column}"
            )

    return dataset


