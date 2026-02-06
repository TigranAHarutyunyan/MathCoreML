from __future__ import annotations

from MathCoreML.utils.CSVStore import CSVStore


class Heart_Disease(CSVStore):
    def __init__(self,CsvPath: str):
        super().__init__(CsvPath)
