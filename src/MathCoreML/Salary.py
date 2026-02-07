from mathcoreml.utils.csvstore import csvstore

class salary(csvstore):
    def __init__(self, csv_path):
        super().__init__(csv_path)

