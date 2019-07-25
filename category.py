catdata = []
catdata.extend([
    {'id': 92, 'name': 'banner', 'supercategory': 'textile'},
    {'id': 93, 'name': 'blanket', 'supercategory': 'textile'},
    {'id': 94, 'name': 'branch', 'supercategory': 'plant'},
    {'id': 95, 'name': 'bridge', 'supercategory': 'building'},
    {'id': 96, 'name': 'building-other', 'supercategory': 'building'},
    {'id': 97, 'name': 'bush', 'supercategory': 'plant'},
    {'id': 98, 'name': 'cabinet', 'supercategory': 'furniture-stuff'},
    {'id': 99, 'name': 'cage', 'supercategory': 'structural'},
    {'id': 100, 'name': 'cardboard', 'supercategory': 'raw-material'},
    {'id': 101, 'name': 'carpet', 'supercategory': 'floor'},
    {'id': 102, 'name': 'ceiling-other', 'supercategory': 'ceiling'},
    {'id': 103, 'name': 'ceiling-tile', 'supercategory': 'ceiling'},
    {'id': 104, 'name': 'cloth', 'supercategory': 'textile'},
    {'id': 105, 'name': 'clothes', 'supercategory': 'textile'},
    {'id': 106, 'name': 'clouds', 'supercategory': 'sky'},
    {'id': 107, 'name': 'counter', 'supercategory': 'furniture-stuff'},
    {'id': 108, 'name': 'cupboard', 'supercategory': 'furniture-stuff'},
    {'id': 109, 'name': 'curtain', 'supercategory': 'textile'},
    {'id': 110, 'name': 'desk-stuff', 'supercategory': 'furniture-stuff'},
    {'id': 111, 'name': 'dirt', 'supercategory': 'ground'},
    {'id': 112, 'name': 'door-stuff', 'supercategory': 'furniture-stuff'},
    {'id': 113, 'name': 'fence', 'supercategory': 'structural'},
    {'id': 114, 'name': 'floor-marble', 'supercategory': 'floor'},
    {'id': 115, 'name': 'floor-other', 'supercategory': 'floor'},
    {'id': 116, 'name': 'floor-stone', 'supercategory': 'floor'},
    {'id': 117, 'name': 'floor-tile', 'supercategory': 'floor'},
    {'id': 118, 'name': 'floor-wood', 'supercategory': 'floor'},
    {'id': 119, 'name': 'flower', 'supercategory': 'plant'},
    {'id': 120, 'name': 'fog', 'supercategory': 'water'},
    {'id': 121, 'name': 'food-other', 'supercategory': 'food-stuff'},
    {'id': 122, 'name': 'fruit', 'supercategory': 'food-stuff'},
    {'id': 123, 'name': 'furniture-other', 'supercategory': 'furniture-stuff'},
    {'id': 124, 'name': 'grass', 'supercategory': 'plant'},
    {'id': 125, 'name': 'gravel', 'supercategory': 'ground'},
    {'id': 126, 'name': 'ground-other', 'supercategory': 'ground'},
    {'id': 127, 'name': 'hill', 'supercategory': 'solid'},
    {'id': 128, 'name': 'house', 'supercategory': 'building'},
    {'id': 129, 'name': 'leaves', 'supercategory': 'plant'},
    {'id': 130, 'name': 'light', 'supercategory': 'furniture-stuff'},
    {'id': 131, 'name': 'mat', 'supercategory': 'textile'},
    {'id': 132, 'name': 'metal', 'supercategory': 'raw-material'},
    {'id': 133, 'name': 'mirror-stuff', 'supercategory': 'furniture-stuff'},
    {'id': 134, 'name': 'moss', 'supercategory': 'plant'},
    {'id': 135, 'name': 'mountain', 'supercategory': 'solid'},
    {'id': 136, 'name': 'mud', 'supercategory': 'ground'},
    {'id': 137, 'name': 'napkin', 'supercategory': 'textile'},
    {'id': 138, 'name': 'net', 'supercategory': 'structural'},
    {'id': 139, 'name': 'paper', 'supercategory': 'raw-material'},
    {'id': 140, 'name': 'pavement', 'supercategory': 'ground'},
    {'id': 141, 'name': 'pillow', 'supercategory': 'textile'},
    {'id': 142, 'name': 'plant-other', 'supercategory': 'plant'},
    {'id': 143, 'name': 'plastic', 'supercategory': 'raw-material'},
    {'id': 144, 'name': 'platform', 'supercategory': 'ground'},
    {'id': 145, 'name': 'playingfield', 'supercategory': 'ground'},
    {'id': 146, 'name': 'railing', 'supercategory': 'structural'},
    {'id': 147, 'name': 'railroad', 'supercategory': 'ground'},
    {'id': 148, 'name': 'river', 'supercategory': 'water'},
    {'id': 149, 'name': 'road', 'supercategory': 'ground'},
    {'id': 150, 'name': 'rock', 'supercategory': 'solid'},
    {'id': 151, 'name': 'roof', 'supercategory': 'building'},
    {'id': 152, 'name': 'rug', 'supercategory': 'textile'},
    {'id': 153, 'name': 'salad', 'supercategory': 'food-stuff'},
    {'id': 154, 'name': 'sand', 'supercategory': 'ground'},
    {'id': 155, 'name': 'sea', 'supercategory': 'water'},
    {'id': 156, 'name': 'shelf', 'supercategory': 'furniture-stuff'},
    {'id': 157, 'name': 'sky-other', 'supercategory': 'sky'},
    {'id': 158, 'name': 'skyscraper', 'supercategory': 'building'},
    {'id': 159, 'name': 'snow', 'supercategory': 'ground'},
    {'id': 160, 'name': 'solid-other', 'supercategory': 'solid'},
    {'id': 161, 'name': 'stairs', 'supercategory': 'furniture-stuff'},
    {'id': 162, 'name': 'stone', 'supercategory': 'solid'},
    {'id': 163, 'name': 'straw', 'supercategory': 'plant'},
    {'id': 164, 'name': 'structural-other', 'supercategory': 'structural'},
    {'id': 165, 'name': 'table', 'supercategory': 'furniture-stuff'},
    {'id': 166, 'name': 'tent', 'supercategory': 'building'},
    {'id': 167, 'name': 'textile-other', 'supercategory': 'textile'},
    {'id': 168, 'name': 'towel', 'supercategory': 'textile'},
    {'id': 169, 'name': 'tree', 'supercategory': 'plant'},
    {'id': 170, 'name': 'vegetable', 'supercategory': 'food-stuff'},
    {'id': 171, 'name': 'wall-brick', 'supercategory': 'wall'},
    {'id': 172, 'name': 'wall-concrete', 'supercategory': 'wall'},
    {'id': 173, 'name': 'wall-other', 'supercategory': 'wall'},
    {'id': 174, 'name': 'wall-panel', 'supercategory': 'wall'},
    {'id': 175, 'name': 'wall-stone', 'supercategory': 'wall'},
    {'id': 176, 'name': 'wall-tile', 'supercategory': 'wall'},
    {'id': 177, 'name': 'wall-wood', 'supercategory': 'wall'},
    {'id': 178, 'name': 'water-other', 'supercategory': 'water'},
    {'id': 179, 'name': 'waterdrops', 'supercategory': 'water'},
    {'id': 180, 'name': 'window-blind', 'supercategory': 'window'},
    {'id': 181, 'name': 'window-other', 'supercategory': 'window'},
    {'id': 182, 'name': 'wood', 'supercategory': 'solid'}
])
catdata.extend([{'id': 183, 'name': 'other', 'supercategory': 'other'}])