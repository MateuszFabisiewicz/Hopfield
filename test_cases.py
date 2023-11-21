
import pandas as pd
import re

class TestCase:
    def __init__(self, filename, dir):
        data = pd.read_csv(dir +"/"+ filename, header=None)
        self.name = filename
        self.patterns = data.to_numpy()
        match = re.search(r'-(\d+)x(\d+)', filename)
        if match:
            self.shape = (int(match.group(2)), (int(match.group(1))))
        else:
            raise ValueError("Invalid file name.")
