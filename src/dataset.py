from __future__ import annotations

from mathcoreml.utils.csvstore import csvstore


class dataset(csvstore):
    def __init__(self,CsvPath: str):
        super().__init__(CsvPath)