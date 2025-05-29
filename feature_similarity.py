import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy as scipy_entropy


def normalize_vector(vec):
    """
    Normaliza um vetor para ter norma (comprimento) igual a 1.
    Retorna o vetor original se a norma for zero.
    """
    vec = np.asarray(vec)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def normalize_distribution(vec, epsilon=1e-10):
    """
    Normaliza um vetor para que a soma de seus elementos seja 1,
    garantindo que todos os valores sejam positivos (>= epsilon).
    Útil para transformar vetores em distribuições de probabilidade.
    """
    vec = np.asarray(vec, dtype=np.float64)
    vec = np.clip(vec, epsilon, np.inf)
    return vec / np.sum(vec)

def gini_index(probabilities: np.ndarray) -> float:
    """
    Calcula o índice de Gini de uma distribuição de probabilidade.
    Mede a desigualdade dos valores (quanto mais próximo de 1, mais desigual).
    """
    return 1 - np.sum(np.square(probabilities))

def entropy_safe(probabilities: np.ndarray) -> float:
    """
    Calcula a entropia de Shannon de uma distribuição de probabilidade,
    protegendo contra log(0) ao limitar os valores mínimos.
    """
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probabilities * np.log(probabilities))


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calcula a divergência de Kullback-Leibler entre duas distribuições de probabilidade.
    Mede o quanto uma distribuição p diverge de uma distribuição q.
    """
    p = normalize_distribution(p, epsilon)
    q = normalize_distribution(q, epsilon)
    return np.sum(p * np.log(p / q))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a similaridade do cosseno entre dois vetores.
    Mede o ângulo entre eles (1 = iguais, 0 = ortogonais, -1 = opostos).
    """
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape for cosine similarity.")
    dot_product = np.dot(x, y)
    return dot_product / (np.linalg.norm(x) * np.linalg.norm(y))


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: int = 2) -> float:
    """
    Calcula a distância de Minkowski de ordem p entre dois vetores.
    Para p=2, equivale à distância Euclidiana.
    """
    return np.power(np.sum(np.power(np.abs(x - y), p)), 1/p)


def lorentzian_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distância Lorentziana entre dois vetores.
    Usa log(1 + |x_i - y_i|) para cada elemento.
    """
    return np.sum(np.log1p(np.abs(x - y)))


def canberra_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distância de Canberra entre dois vetores.
    É sensível a pequenas diferenças próximas de zero.
    """
    return distance.canberra(x, y)


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distância do cosseno entre dois vetores.
    É 1 - similaridade do cosseno.
    """
    return 1 - cosine_similarity(x, y)


def hellinger_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distância de Hellinger entre duas distribuições de probabilidade.
    Mede a similaridade entre distribuições (sempre >= 0).
    """
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Hellinger distance requires non-negative vectors.")
    x = normalize_distribution(x)
    y = normalize_distribution(y)
    return np.sqrt(2 * np.sum((np.sqrt(x) - np.sqrt(y)) ** 2))


def squared_chi_squared(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distância qui-quadrado ao quadrado entre dois vetores.
    Útil para comparar histogramas ou distribuições.
    """
    return np.sum((x - y) ** 2 / (x + y + 1e-8))


def jensen_shannon_divergence(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a divergência de Jensen-Shannon entre duas distribuições de probabilidade.
    É uma versão simétrica e suavizada da divergência KL.
    """
    x = normalize_distribution(x)
    y = normalize_distribution(y)
    m = 0.5 * (x + y)
    return 0.5 * (scipy_entropy(x, m) + scipy_entropy(y, m))


def vicis_symmetric(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distância simétrica de Vicis entre dois vetores.
    Usa o quadrado da diferença dividido pelo quadrado do menor valor.
    """
    return np.sum((x - y) ** 2 / (np.minimum(x, y) ** 2 + 1e-8))


def hassanat_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distância de Hassanat entre dois vetores.
    Mede a similaridade considerando valores mínimos e máximos de cada posição.
    """
    min_vals = np.minimum(x, y)
    max_vals = np.maximum(x, y)

    mask = min_vals >= 0
    numerator = 1 + min_vals
    denominator = 1 + max_vals

    numerator[~mask] += np.abs(min_vals[~mask])
    denominator[~mask] += np.abs(min_vals[~mask])

    return 1 - np.mean(numerator / denominator)


def entropic_distance(prob_dist1: np.ndarray, prob_dist2: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calcula uma distância baseada na diferença absoluta entre índices de Gini e entropias
    de duas distribuições de probabilidade.
    """
    prob_dist1 = normalize_distribution(prob_dist1, epsilon)
    prob_dist2 = normalize_distribution(prob_dist2, epsilon)

    gini1 = gini_index(prob_dist1)
    gini2 = gini_index(prob_dist2)

    entropy1 = entropy_safe(prob_dist1)
    entropy2 = entropy_safe(prob_dist2)

    return abs(gini1 - gini2) + abs(entropy1 - entropy2)