import numpy as np
import tenseal as ts

def check_feature_vector(vector, ensure_2d=True, dtype="infer_float", ensure_all_finite=True):
    """
    Validates the feature vector.
    Expected shape: (1, 384)
    """
    if not isinstance(vector, np.ndarray):
        raise ValueError("[Validation] Feature vector must be a numpy array.")

    if ensure_2d:
        if vector.ndim != 2 or vector.shape[0] != 1 or vector.shape[1] != 384:
            raise ValueError(
                "[Validation] Feature vector must be of shape (1, 384).")

    if ensure_all_finite and not np.all(np.isfinite(vector)):
        raise ValueError(
            "[Validation] Feature vector contains non-finite values (NaN or Inf).")

    if dtype == "infer_float":
        dtype = np.float32 if np.issubdtype(
            vector.dtype, np.floating) else np.float64
    elif not np.issubdtype(vector.dtype, np.floating):
        raise ValueError("[Validation] Feature vector must be of float type.")

    print("[Validation] Feature vector validation complete.")
    return vector.astype(dtype)


def normalize_vector(vector):
    """
    Normalizes the vector to unit length.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero

    print("[Normalization] Feature vector normalized.")
    return vector / norm



# Initialize TenSEAL context for CKKS scheme with secret key (for demo purposes)
print("[TenSEAL] Initializing TenSEAL context with secret key...")
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2**40
context.generate_galois_keys()
print("[TenSEAL] Context initialized. (Secret key is present)")


def encrypt_feature_vector(features):
    """
    Encrypts the provided feature vector using TenSEAL.
    The input 'features' should be a (1,384) numpy array.
    """
    print("[Encryption] Encrypting feature vector...")
    # Flatten the features from (1, 384) to (384,)
    encrypted = ts.ckks_vector(context, features.flatten().tolist())
    print("[Encryption] Feature vector encrypted.")
    return encrypted

def encrypt_embedding_for_neo4j(embedding):
    """
    Accepts an embedding (NumPy array of shape (1, 384)), checks and normalizes it,
    encrypts it using TenSEAL, and returns the serialized encrypted embedding.
    This serialized encrypted data is ready to be stored in Neo4j.
    """
    print("[Django FHE] Received embedding for encryption.")
    checked_embedding = check_feature_vector(embedding)
    normalized_embedding = normalize_vector(checked_embedding)
    encrypted_vector = encrypt_feature_vector(normalized_embedding)
    serialized_encrypted = encrypted_vector.serialize()
    print("[Django FHE] Embedding encrypted and serialized for storage.")
    return serialized_encrypted
i