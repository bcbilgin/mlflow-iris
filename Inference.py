import logging

import fire
import pandas as pd
from joblib import load


def main(classifier_path, input_path):
    # Load classifier
    logging.info(f"loading classifier {classifier_path}")
    classifier = load(classifier_path)
    # Load input data
    logging.info(f"loading input file {input_path}")
    data = pd.read_csv(input_path)
    # Infer class
    logging.info(f"predicting class")
    predicted = classifier.predict(data)
    print(predicted)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)
