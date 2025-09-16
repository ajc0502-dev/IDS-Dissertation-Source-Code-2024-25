# Import modules
import argparse

from src.train_dt import train_decision_tree
from src.train_ensemble import train_ensemble
from src.train_cnn import train_cnn
from src.predict import predict_new_data


def main(): # Parses areguments to CLI - Need to segregate at some point
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dt", "ensemble", "cnn"], required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str)
    parser.add_argument("--data", type=str)

    parser.add_argument("--max_depth", type=int, default=5)

    parser.add_argument("--tfidf_max_features", type=int, default=20000)
    parser.add_argument("--tfidf_ngram_max", type=int, default=2)
    parser.add_argument("--tfidf_min_df", type=int, default=2)
    parser.add_argument("--rf_estimators", type=int, default=300)

    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()

    if args.train:
        if args.model == "dt":
            train_decision_tree(data_path=args.data, max_depth=args.max_depth) # Parse params for DT

        elif args.model == "ensemble": # Parse params for ensemble
            train_ensemble(
                data_path=args.data,
                tfidf_max_features=args.tfidf_max_features,
                tfidf_ngram_max=args.tfidf_ngram_max,
                tfidf_min_df=args.tfidf_min_df,
                rf_estimators=args.rf_estimators
            )

        elif args.model == "cnn": # Parse params for CNN
            train_cnn(
                data_path=args.data,
                max_len=args.max_len,
                vocab_size=args.vocab_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                dropout=args.dropout
            )
        return

    if args.predict: # Parse args for data retraining
        preds = predict_new_data(args.predict, model_name=args.model)
        print(preds)
        return

    parser.error("Specify --train or --predict")

if __name__ == "__main__":
    main()
