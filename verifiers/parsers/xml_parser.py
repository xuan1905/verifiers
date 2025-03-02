import re
from typing import List, Dict, Any, Union, Tuple, Optional
from types import SimpleNamespace

class XMLParser:
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]]):
        """
        Initialize the parser with field definitions.
        
        Each field may be:
          - a string (e.g. "reasoning"): the XML tag is fixed.
          - a tuple of alternatives (e.g. ("code", "answer")): the first element is
            the canonical name used for formatting, and all elements are allowed tags
            when parsing.
            
        The schema is assumed to have no duplicate names.
        """
        self._fields: List[Tuple[str, List[str]]] = []  # List of (canonical, [alternatives])
        seen = set()
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                if not all(isinstance(alt, str) for alt in field):
                    raise TypeError("All alternatives in a tuple must be strings.")
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))
    
    def get_fields(self) -> List[str]:
        """Return a list of the canonical field names (in order)."""
        return [canonical for canonical, _ in self._fields]
    
    def format(self, **kwargs) -> str:
        """
        Format the provided keyword arguments into an XML string.
        
        For fields with alternatives (tuple), the canonical name (the first element)
        is used as the XML tag. The method looks for a provided value using any of the
        allowed names (preferring the canonical if present).
        
        Example usage:
            parser = XMLParser(['reasoning', ('code', 'answer')])
            formatted_str = parser.format(reasoning="...", code="...")
        """
        parts = []
        for canonical, alternatives in self._fields:
            value = None
            # Look for a provided value using any of the acceptable keys,
            # preferring the canonical name if it exists.
            if canonical in kwargs:
                value = kwargs[canonical]
            else:
                for alt in alternatives:
                    if alt in kwargs:
                        value = kwargs[alt]
                        break
            if value is None:
                raise ValueError(f"Missing value for field '{canonical}' (allowed: {alternatives})")
            # Use the canonical name as the tag for formatting.
            parts.append(f"<{canonical}>\n{value}\n</{canonical}>")
        return "\n".join(parts)
    
    def parse(self, text: str, strip: bool = True) -> Any:
        """
        Parse the given XML string and return an object with attributes corresponding
        to all allowed tags in the schema.
        
        For each field defined:
          - If it is a simple field (e.g. 'reasoning'), the output object will have
            an attribute 'reasoning' set to the text content (or None if missing).
          - If it is defined with alternatives (e.g. ("code", "answer")), the output
            object will have attributes for *each* allowed tag name. For example,
            if the schema is ['reasoning', ('code', 'answer')], then both
            `result.code` and `result.answer` are always accessible. If a tag is not
            found in the XML, its corresponding attribute is set to None.
        """
        results: Dict[str, Optional[str]] = {}
        for canonical, alternatives in self._fields:
            # For each allowed alternative tag, search independently.
            for alt in alternatives:
                # Regex pattern to capture the content between the tags.
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    results[alt] = match.group(1).strip() if strip else match.group(1)
                else:
                    results[alt] = None
        return SimpleNamespace(**results)