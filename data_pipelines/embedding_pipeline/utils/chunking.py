import tiktoken
import uuid
import hashlib


def chunk_text_by_tokens(
    tokenizer: tiktoken.core.Encoding, text: str, max_tokens: int = 256
):
    """
    Splits text into chunks based on token count using a tokenizer.

    This function takes input text and divides it into smaller chunks where each chunk
    contains at most the specified number of tokens. The chunking is performed by first
    encoding the text into tokens, then splitting the token sequence into segments,
    and finally decoding each segment back into text.

    Parameters:
        tokenizer (tiktoken.core.Encoding): The tokenizer object used to encode and 
            decode text into tokens.
        text (str): The input text to be chunked.
        max_tokens (int, optional): The maximum number of tokens per chunk. 
            Defaults to 256.

    Returns:
        list[str]: A list of text chunks, where each chunk contains at most 
            max_tokens tokens when encoded with the provided tokenizer.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


def chunk_text_with_overlap(
    tokenizer: tiktoken.core.Encoding, 
    text: str, 
    max_tokens=256, 
    overlap=32
):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start += max_tokens - overlap
    return chunks


def generate_chunk_id(url: str, chunk_text: str) -> str:
    """
    Generate a deterministic UUID based on URL + chunk text.
    Ensures same content always yields same ID.
    """
    hash_input = (url + chunk_text).encode("utf-8")
    hash_bytes = hashlib.sha1(hash_input).digest()  # or sha256 for stronger hashing
    return str(uuid.UUID(bytes=hash_bytes[:16]))