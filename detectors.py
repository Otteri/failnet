# Different failure detection implementations for various cases

def compareSingleValue(actual, pred) -> bool:
    """
    Compares predictions and measurement values from a single scan

    Args:
        pred (1d array): Prections made by the NN
        actual (1d array): Measured values
    Returns:
        boolean which tells if threshold was exceeded and if there is an issue.
    """
    epsilon = 2.0
    is_failure = False

    assert len(actual) == len(pred), "Dimension are different, How!?!"

    for i in range(0, len(actual)):
        diff = abs(actual[i] - pred[i])
        print("diff: ", diff)
        if diff > epsilon:
            is_failure = True

    return is_failure
