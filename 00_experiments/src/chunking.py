from langchain_text_splitters import TextSplitter
from langchain_text_splitters.character import _split_text_with_regex
from langchain_core.documents import Document
from typing import Any, Literal
import re


class SentenceTextSplitter(TextSplitter):
    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [".", "。", "!", "?", "！", "？"]
        self._is_separator_regex = is_separator_regex

        self._abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'phd', 'md', 'esq',
            'vs', 'etc', 'inc', 'ltd', 'co', 'corp', 'dept', 'govt', 'org',
            'no', 'nos', 'vol', 'pp', 'fig', 'figs', 'ch', 'sec', 'art',
            'i.e', 'e.g', 'cf', 'ca', 'circa', 'et', 'al', 'ibid', 'op', 'cit',
            'st', 'ave', 'blvd', 'rd', 'dr', 'ln', 'ct', 'pl', 'sq',
            'a.m', 'p.m', 'am', 'pm', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
            'aug', 'sep', 'oct', 'nov', 'dec', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
            'max', 'min', 'approx', 'est', 'ref', 'ed', 'eds', 'rev'
        }

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            separator_ = _s if self._is_separator_regex else re.escape(_s)
            if not _s:
                separator = _s
                break
            if re.search(separator_, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        separator_ = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, separator_, keep_separator=self._keep_separator
        )

        for s in splits:
            if not new_separators:
                final_chunks.append(s)
            else:
                other_info = self._split_text(s, new_separators)
                final_chunks.extend(other_info)
        return final_chunks

    def _protect_abbrev(self, text: str) -> str:
        protected_text = text
        
        # First, protect common initial patterns (e.g., "P. G.", "J. K.", "A. B. C.")
        # Match single letters followed by periods, potentially repeated
        initial_pattern = re.compile(r'\b([A-Z]\.\s*)+[A-Z]\.?(?=\s|$|[,;:])', re.IGNORECASE)
        protected_text = initial_pattern.sub(lambda m: m.group(0).replace('.', '§PERIOD§'), protected_text)
        
        # Then handle regular abbreviations
        for abbrev in self._abbreviations:
            # Handle abbreviations that already end with a period
            if abbrev.endswith('.'):
                base_abbrev = abbrev[:-1]
                pattern = re.compile(rf'\b{re.escape(base_abbrev)}\.(?=[,;:]*\s*[A-Z0-9])', re.IGNORECASE)
            else:
                # Handle abbreviations without period - match both with and without trailing period
                pattern = re.compile(rf'\b{re.escape(abbrev)}\.?(?=[,;:]*\s*[A-Z0-9])', re.IGNORECASE)
            
            protected_text = pattern.sub(lambda m: m.group(0).replace('.', '§PERIOD§'), protected_text)
    
        # Protect decimal numbers
        protected_text = re.sub(r'\b\d+\.\d+\b', lambda m: m.group(0).replace('.', '§PERIOD§'), protected_text)
        return protected_text
    
    def split_text(self, text: str) -> list[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text: The input text to be split.

        Returns:
            A list of text chunks obtained after splitting.
        """
        protected_text = self._protect_abbrev(text)
        chunks = self._split_text(protected_text, self._separators)
        chunks = [chunk.replace('§PERIOD§', '.') for chunk in chunks]
        return [chunk.strip() for chunk in chunks if chunk.strip()]