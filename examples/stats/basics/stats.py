from math import sqrt


def accept_sequence(func):
    """Decorator to allow either *args or single sequence as input."""

    def wrapper(*args):
        if len(args) == 1:
            return func(*args[0])
        return func(*args)

    return wrapper


@accept_sequence
def mean(*args):
    return sum(args) / len(args)


@accept_sequence
def median(*args):
    args = sorted(args)
    if len(args) % 2 == 0:
        i = round((len(args) + 1) / 2)
        j = i - 1
        return (args[i] + args[j]) / 2
    else:
        k = round(len(args) / 2)
        return args[k]


@accept_sequence
def mode(*args):
    """Calculates the mode (number that occurs most often) of a list of numbers."""
    counts = {a: args.count(a) for a in args}
    max_list = [k for k, v in counts.items() if v == max(counts.values())]
    return max_list


@accept_sequence
def variance(*args, sample=True):
    """Calculates the variance (how data is spread around the mean) of a list of numbers."""
    return (
        0
        if len(args) == 1
        else sum([(x - mean(*args)) ** 2 for x in args]) / (len(args) - int(sample))
    )


def standard_deviation(*args):
    """Calculates the standard deviation (the square root of the variance) of a list of numbers."""
    return sqrt(variance(*args))


def coefficient_of_variation(*args):
    """Calculates the coefficient of variation (the ratio of the standard deviation to the mean) of a list of numbers."""
    return standard_deviation(*args) / mean(*args)


def covariance(a: list, b: list):
    """
    Calculates the covariance (telling us if two datasets are moving in the same direction) of a list of numbers.
    > 0 : Moving Together
    < 0: Moving Opposite
    = 0: Independent
    """
    assert len(a) == len(b)
    return sum([(x - mean(a)) * (y - mean(b)) for x, y in zip(a, b)]) / (len(a) - 1)


def correlation_coefficient(a: list, b: list):
    return covariance(a, b) / (standard_deviation(a) * standard_deviation(b))


@accept_sequence
def sample_error(*args):
    """Calculates the sample error (aka Standard error of the mean (SEM)) of a list of numbers."""
    return standard_deviation(*args) / sqrt(len(args))


@accept_sequence
def minmax_normalise(*args):
    """Normalises a list of numbers to a range of 0 to 1."""
    return [(x - min(args)) / (max(args) - min(args)) for x in args]


@accept_sequence
def zscore_normalise(*args):
    """Normalises a list of numbers to a range of -1 to 1."""
    return [(x - mean(args)) / standard_deviation(args) for x in args]


if __name__ == "__main__":
    ##############################################################
    a = [1, 2, 3, 4, 5, 6]
    b = [100, 2, 3, 4, 200]
    print(f"Mean: {a} == {mean(a)}")
    print(f"Mean: {b} == {mean(b)}")
    print(f"Mean: {1, 2, 3} == {mean(1, 2, 3)}")

    print(f"Median: {a} == {median(a)}")
    print(f"Median: {b} == {median(b)}")
    print(f"Median: {1, 2, 3} == {median(1, 2, 3)}")

    ##############################################################
    a = [1, 1, 1, 2]
    b = [1, 1, 2, 2]
    print(f"Mode: {a} == {mode(a)}")
    print(f"Mode: {b} == {mode(b)}")
    print(f"Mode: {1, 2, 3} == {mode(1, 2, 3)}")

    ##############################################################
    a = [4, 6, 3, 5, 2]
    b = [1, 1, 2, 2]
    print(f"Variance: {a} == {variance(a)}")
    print(f"Variance: {b} == {variance(b)}")
    print(f"Variance: {1, 2, 3} == {variance(1, 2, 3)}")

    # Large SD = Large Spread
    print(f"SD: {a} == {standard_deviation(a)}")
    print(f"SD: {b} == {standard_deviation(b)}")
    print(f"SD: {1, 2, 3} == {standard_deviation(1, 2, 3)}")

    ##############################################################
    miles = [3, 4, 4.5, 3.5]
    kilometers = [4.828, 6.437, 7.242, 5.632]

    # Compares two measurements that operate on different scales (i.e. Miles vs. Kilometers).
    # By dividing by the mean, we can see they have the same dispersion.
    print(f"Coef. Variation KM: {miles} == {coefficient_of_variation(miles)}")
    print(
        f"Coef. Variation Miles: {kilometers} == {coefficient_of_variation(kilometers)}"
    )

    ##############################################################
    companies = ["Apple", "Google", "Microsoft", "Amazon", "Twitter"]
    market_cap = [1532, 1488, 1343, 928, 615]
    earnings = [58, 35, 75, 41, 17]

    # Both covariance and correlation measure the relationship and the dependency between two variables.
    # Covariance indicates the direction of the linear relationship between variables.
    # Correlation measures both the strength and direction of the linear relationship between two variables.
    # Correlation values are standardized.
    # Covariance values are not standardized.
    print(f"Covariance: {covariance(market_cap, earnings)}")
    print(f"Correlation: {correlation_coefficient(market_cap, earnings)}")

    ##############################################################
    # Central Limit Theorem
    # More Samples = More Likely to be Normally Distributed
    # More Samples gets you closer to the trye Mean (and data more likely to be Normally Distributed).
    # As the sample size increases the standard deviation decreases.

    # Standard / Sample Error (Accuracy Measure of an estimate)
    # Measures how much discrepancy there is likely to be in a sample's mean compared to the population mean. The SEM takes the SD and divides it by the square root of the sample size.

    ##############################################################
