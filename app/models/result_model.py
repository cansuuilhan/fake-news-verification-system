from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AnalysisResult:
    label: str
    confidence: float
    explanation: str
    claims: List[str] = field(default_factory=list)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    verifications: Dict[str, dict] = field(default_factory=dict)