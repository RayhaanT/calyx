import json
import sys
import random

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        d = json.load(f)
    width = di = None
    for k, v in d.items():
        if k.startswith("tdi"):
            if width is None:
                width = v["format"]["width"]
                di = [random.randint(0, (1<<width)-1) for _ in v["data"]]
            v["format"] = {"is_signed":False,"numeric_type":"bitnum","width":width}
            v["data"] = di
            di = [(i+1) % (1<<width) for i in di]
    with open(sys.argv[1], "w") as f:
        json.dump(d, f)
