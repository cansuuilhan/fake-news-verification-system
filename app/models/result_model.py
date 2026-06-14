from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AnalysisResult:
    label: str
    confidence: float
    explanation: str

    summary: str = ""

    final_decision: str = ""
    final_score: float = 0.0

    claims: List[str] = field(default_factory=list)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    verifications: Dict[str, dict] = field(default_factory=dict)