import datasets

import nalp.utils.preprocess as p


def bleu_score(preds, refs, n_grams=1, smoothing=False):
    """Calculates the BLEU score based on predictions and references.

    Args:
        preds (list): List of predictions.
        refs (list): List of lists of references.
        n_grams (int): Maximum n-grams to be considered.
        smoothing (bool): Whether to apply smoothing.
        
    Returns:
        Dictionary holding the BLEU score and meta-information.

    """

    # Prepare the predictions and references by performing a simple tokenization
    tokenized_preds = [p.tokenize_to_word(pred) for pred in preds]
    tokenized_refs = [[p.tokenize_to_word(r) for r in ref] for ref in refs]

    # Checks the length between both `tokenized_preds` and `tokenized_refs`
    if len(tokenized_preds) != len(tokenized_refs):
        raise Exception('There should be at least one reference for each prediction.')
    
    # Loads the BLEU metric
    bleu = datasets.load_metric('bleu')

    # Computes the BLEU score
    score = bleu.compute(predictions=tokenized_preds, references=tokenized_refs,
                         max_order=n_grams, smooth=smoothing)

    return score


def meteor_score(preds, refs, alpha=0.9, beta=3, gamma=0.5):
    """Calculates the METEOR score based on predictions and references.

    Args:
        preds (list): List of predictions.
        refs (list): List of references.
        alpha (float): Controls relative weights of precision and recall.
        beta (int): Controls the shape of penalty function.
        gamma (float): Controls the relative weight assigned to fragmentation penalty.
        
    Returns:
        Dictionary holding the METEOR score and meta-information.

    """
    
    # Loads the METEOR metric
    meteor = datasets.load_metric('meteor')

    # Checks the length between both `preds` and `refs`
    if len(preds) != len(refs):
        raise Exception('There should be a single reference for each prediction.')

    # Computes the METEOR score
    score = meteor.compute(predictions=preds, references=refs,
                           alpha=alpha, beta=beta, gamma=gamma)

    return score


def rouge_score(preds, refs):
    """Calculates the ROUGE score based on predictions and references.

    Args:
        preds (list): List of predictions.
        refs (list): List of references.
        
    Returns:
        Dictionary holding the ROUGE score and meta-information.

    """
    
    # Loads the ROUGE metric
    rouge = datasets.load_metric('rouge')

    # Checks the length between both `preds` and `refs`
    if len(preds) != len(refs):
        raise Exception('There should be a single reference for each prediction.')

    # Computes the ROUGE score
    score = rouge.compute(predictions=preds, references=refs)

    return score
