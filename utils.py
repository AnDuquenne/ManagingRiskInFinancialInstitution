import scipy.stats as stats
import numpy as np

def calculate_var_es(mean, variance, confidence_level):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for a Gaussian distribution.

    Parameters:
        mean (float): Mean of the Gaussian distribution.
        variance (float): Variance of the Gaussian distribution.
        confidence_level (float): Confidence level (e.g., 0.95 for 95%).

    Returns:
        tuple: VaR and ES values.
    """
    # Calculate the standard deviation
    std_dev = np.sqrt(variance)

    # Convert confidence level to the corresponding quantile
    alpha = 1 - confidence_level
    z_alpha = stats.norm.ppf(alpha)  # Critical value (inverse CDF)

    # Value at Risk (VaR)
    var = mean + z_alpha * std_dev

    # Expected Shortfall (ES)
    # Using the formula ES = mean + std_dev * (PDF(alpha) / CDF(alpha))
    pdf_alpha = stats.norm.pdf(z_alpha)  # PDF at the critical value
    cdf_alpha = alpha  # CDF at the critical value is alpha
    es = mean - (std_dev * pdf_alpha / cdf_alpha)

    return var, es

def cornish_fisher_var_es(mean, variance, confidence_level, skewness, kurtosis):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for a Gaussian distribution.

    Parameters:
        mean (float): Mean of the Gaussian distribution.
        variance (float): Variance of the Gaussian distribution.
        confidence_level (float): Confidence level (e.g., 0.95 for 95%).
        skewness (float): Skewness of the distribution.
        kurtosis (float): Kurtosis of the distribution.

    Returns:
        tuple: VaR and ES values.
    """
    # Calculate the standard deviation
    std_dev = np.sqrt(variance)

    # Convert confidence level to the corresponding quantile
    alpha = 1 - confidence_level
    z_alpha = stats.norm.ppf(alpha)  # Critical value (inverse CDF)

    # Calculate the modified z-score using Cornish-Fisher expansion
    z_alpha_cf = (z_alpha
                  + (z_alpha**2 - 1) / 6 * skewness
                  + (z_alpha**3 - 3*z_alpha) / 24 * (kurtosis - 3)
                  - 1/36 * (2*z_alpha**3 - 5*z_alpha) * skewness**2)

    # Value at Risk (VaR)
    var = mean + z_alpha_cf * std_dev

    # Expected Shortfall (ES)
    # Using the formula ES = mean + std_dev * (PDF(alpha) / CDF(alpha))
    pdf_alpha = stats.norm.pdf(z_alpha_cf)  # PDF at the critical value
    cdf_alpha = alpha  # CDF at the critical value is alpha
    es = mean - (std_dev * pdf_alpha / cdf_alpha)

    return var, es