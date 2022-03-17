from enum import Enum


class Task(Enum):
    CodeClassification = "CodeClassification"
    VulnerabilityDetection = "VulnerabilityDetection"
    CloneDetection = "CloneDetection"


class TaskType(Enum):
    Classification = "Classification"
    Summarization = "Summarization"
    PairwiseClassification = "PairwiseClassification"
