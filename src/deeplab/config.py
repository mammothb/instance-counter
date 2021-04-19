import numpy as np

from src.deeplab.visualize import label_to_color_image


DL_LABEL_NAMES = np.asarray([
    "others", "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest",
    "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator",
    "grandstand", "path", "stairs", "runway", "case", "pool", "pillow",
    "screen", "stairway", "river", "bridge", "bookcase", "blind", "coffee",
    "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm",
    "kitchen", "computer", "swivel", "boat", "bar", "arcade", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television", "airplane", "dirt", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage",
    "van", "ship", "fountain", "conveyer", "canopy", "washer", "plaything",
    "swimming", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic", "tray",
    "ashcan", "fan", "pier", "crt", "plate", "monitor", "bulletin", "shower",
    "radiator", "glass", "clock", "flag"
])
DL_LABEL_DICT = {
    "sand": [
        "animal", "awning", "bridge", "ceiling", "chair", "column", "curtain",
        "door", "earth", "fence", "floor", "grass", "land", "palm", "pier",
        "plant", "railing", "rock", "stairs", "stairway", "swivel", "table",
        "tank", "tent", "tree", "towel", "tower", "wall", "water"
    ],
    "sea": ["airplane", "boat", "building", "hill", "ship", "swimming"]
}
DL_LABEL_MERGE = np.arange(len(DL_LABEL_NAMES))
for key in DL_LABEL_DICT:
    key_index = np.where(DL_LABEL_NAMES == key)
    for label in DL_LABEL_DICT[key]:
        DL_LABEL_MERGE[np.where(DL_LABEL_NAMES == label)] = key_index
# DL_LABEL_MAP = np.arange(len(DL_LABEL_NAMES)).reshape(len(DL_LABEL_NAMES), 1)
# DL_COLOR_MAP = label_to_color_image(DL_LABEL_MAP, DL_LABEL_MERGE)
