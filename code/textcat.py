#!/usr/bin/env python3
"""
Classifies text files as either 'gen' or 'spam' using two trained language models and Bayes' Theorem.
"""
import argparse
import logging
import math
from pathlib import Path
import torch
import sys
from probs import LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gen_model",
        type=Path,
        help="path to the trained model for genuine emails",
    )
    parser.add_argument(
        "spam_model",
        type=Path,
        help="path to the trained model for spam emails",
    )
    parser.add_argument(
        "prior_gen",
        type=float,
        help="prior probability of the genuine emails model (p(gen))"
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu', 'cuda', 'mps'],
        help="device to use for PyTorch (cpu, cuda, or mps if on a Mac)"
    )

    # Logging verbosity settings
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob

def classifier (file:Path, gen_model: LanguageModel, spam_model: LanguageModel, threshold: float)->str:
    gen_log_prob = file_log_prob(file,gen_model)
    spam_log_prob = file_log_prob(file,spam_model)
    gen_prior = math.log(threshold)
    spam_prior = math.log(1-threshold)
    
    if (gen_log_prob+gen_prior)>(spam_log_prob+spam_prior):
        return "gen.model"
    else:
        return "spam.model"

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    # Load both models
    log.info("Loading models...")
    gen_model = LanguageModel.load(args.gen_model, device=args.device)
    spam_model = LanguageModel.load(args.spam_model, device=args.device)

    # Sanity check: Ensure both models have the same vocabulary
    if gen_model.vocab != spam_model.vocab:
        log.critical("The vocabularies of the two models are not the same! Cannot proceed.")
        sys.exit(1)


    # Classify each test file based on log-probabilities from both models
    count_gen = 0
    count_spam = 0
    for file in args.test_files:
        classification = classifier(file, gen_model, spam_model, args.prior_gen)
        if classification == 'gen.model':
            count_gen += 1 
            print(f"{args.gen_model.name:<12} {file.name}")
        else:
            count_spam += 1 
            print(f"{args.spam_model.name:<12} {file.name}")
    count_file = count_gen + count_spam
    print(f'{count_gen} files were more probably from {args.gen_model} ({count_gen/count_file*100:.2f}%)')
    print(f'{count_spam} files were more probably from {args.spam_model} ({count_spam/count_file*100:.2f}%)')

if __name__ == "__main__":
    main()