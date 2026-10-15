import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.models.test_model import test_model_with_ranking
from src.models.test_bm25 import test as test_bm25
from src.models_config import (get_model_q_text, get_model_doc_text,
                                get_model_normalize)


def test_ranking(model_dir_path: str = "./output",
                 test_dataset_path: str = "./data/test.jsonl",
                 pooling_mode: str = 'cls'):
    """
    Rank-aware evaluation for
    any sentence transformer model
    """
    print("test ranking model")
    print("model_dir_path: ", model_dir_path)
    print("test_dataset_path: ", test_dataset_path)

    # Get model directory name and fetch config settings
    model_dir_name = os.path.basename(model_dir_path)
    q_text = get_model_q_text(model_dir_name)
    doc_text = get_model_doc_text(model_dir_name)
    normalize = get_model_normalize(model_dir_name)

    # absolute paths
    # test
    test_model_with_ranking(model_dir_path, test_dataset_path, pooling_mode,
                           q_text=q_text, doc_text=doc_text, normalize=normalize)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        type=str,
                        choices=['test-ranking', 'test-bm25'],
                        help='the mode specifies what should be executed [test-ranking, test-bm25]'
                        )
    parser.add_argument("--test-model-path",
                        type=str,
                        default="./pretrained_models/bert_for_patents",
                        help="the path to the model to test"
                        )
    parser.add_argument("--test-dataset-path",
                        type=str,
                        default="./data/patents_to_publications.jsonl",
                        help="the path to the dataset to test"
                        )
    parser.add_argument('--pooling',
                        type=str,
                        default="cls",
                        help='the mode specifies pooling type for testing [cls, mean]'
                        )
    args = parser.parse_args()

    print(args.mode)
    # check mode
    if args.mode == 'test-ranking':
        test_ranking(model_dir_path=args.test_model_path,
                     test_dataset_path=args.test_dataset_path,
                     pooling_mode=args.pooling)
    elif args.mode == 'test-bm25':
        test_bm25(args.test_dataset_path)
    else:
        raise ValueError(
            'Invalid mode, please choose between "test-ranking", "test-bm25"')