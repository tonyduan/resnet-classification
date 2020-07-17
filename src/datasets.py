import numpy as np
import os
import pickle
import torch
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from src.lib.zipdata import ZipData


class PrecisionTransform(object):

    def __init__(self, precision):
        self.precision = precision

    def __call__(self, x):
        if self.precision == "float":
            return x.float()
        if self.precision == "half":
            return x.half()
        if self.precision == "double":
            return x.double()

def get_train_val_split(targets, val_size=1000, seed=12345):
    """
    Return indices of a train/val split, stratified by target.
    """
    np.random.seed(seed)
    idxs = np.arange(len(targets))
    np.random.shuffle(idxs)
    tr_idxs = np.zeros(len(targets) - val_size, dtype=np.int32)
    val_idxs = np.zeros(val_size, dtype=np.int32)
    num_labels = max(targets) + 1
    tr_ptr, val_ptr = 0, 0
    for i in range(num_labels):
        tr_strat = idxs[targets == i][val_size // num_labels:]
        val_strat = idxs[targets == i][:val_size // num_labels]
        tr_idxs[tr_ptr:tr_ptr + len(tr_strat)] = tr_strat
        val_idxs[val_ptr:val_ptr + len(val_strat)] = val_strat
        tr_ptr += len(tr_strat)
        val_ptr += len(val_strat)
    return tr_idxs, val_idxs

def get_dim(name):
    if name.startswith("cifar"):
        return 3 * 32 * 32
    if name == "svhn":
        return 3 * 32 * 32
    if name == "mnist":
        return 28 * 28
    if name == "fashion":
        return 28 * 28
    if name == "imagenet":
        return 3 * 224 * 224
    if name == "ds-imagenet":
        return 3 * 32 * 32

def get_num_labels(name):
    if name == "imagenet" or name == "ds-imagenet":
        return 1000
    if name == "cifar100":
        return 100
    return 10

def get_label_names(name):
    if name == "cifar":
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if name == "cifar100":
        return ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout", "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups", "plates", "apples", "mushrooms", "oranges", "pears", "sweet peppers", "clock", "computer keyboard", "lamp", "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly", "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee", "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail", "spider", "worm", "baby", "boy", "girl", "man", "woman", "crocodile", "dinosaur", "lizard", "snake", "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor"]
    if name == "imagenet" or name == "ds-imagenet":
        return ["kit_fox", "english_setter", "siberian_husky", "australian_terrier", "english_springer", "grey_whale", "lesser_panda", "egyptian_cat", "ibex", "persian_cat", "cougar", "gazelle", "porcupine", "sea_lion", "malamute", "badger", "great_dane", "walker_hound", "welsh_springer_spaniel", "whippet", "scottish_deerhound", "killer_whale", "mink", "african_elephant", "weimaraner", "coated_wheaten_terrier", "dandie_dinmont", "red_wolf", "old_english_sheepdog", "jaguar", "otterhound", "bloodhound", "airedale", "hyena", "meerkat", "giant_schnauzer", "titi", "toed_sloth", "sorrel", "footed_ferret", "dalmatian", "tan_coonhound", "papillon", "skunk", "staffordshire_bullterrier", "mexican_hairless", "bouvier_des_flandres", "weasel", "miniature_poodle", "cardigan", "malinois", "bighorn", "fox_squirrel", "colobus", "tiger_cat", "lhasa", "impala", "coyote", "yorkshire_terrier", "newfoundland", "brown_bear", "red_fox", "norwegian_elkhound", "rottweiler", "hartebeest", "saluki", "grey_fox", "schipperke", "pekinese", "brabancon_griffon", "west_highland_white_terrier", "sealyham_terrier", "guenon", "mongoose", "indri", "tiger", "irish_wolfhound", "wild_boar", "entlebucher", "zebra", "ram", "french_bulldog", "orangutan", "basenji", "leopard", "bernese_mountain_dog", "maltese_dog", "norfolk_terrier", "toy_terrier", "vizsla", "cairn", "squirrel_monkey", "groenendael", "clumber", "siamese_cat", "chimpanzee", "komondor", "afghan_hound", "japanese_spaniel", "proboscis_monkey", "guinea_pig", "white_wolf", "ice_bear", "gorilla", "borzoi", "toy_poodle", "kerry_blue_terrier", "ox", "scotch_terrier", "tibetan_mastiff", "spider_monkey", "doberman", "boston_bull", "greater_swiss_mountain_dog", "appenzeller", "tzu", "irish_water_spaniel", "pomeranian", "bedlington_terrier", "warthog", "arabian_camel", "siamang", "miniature_schnauzer", "collie", "golden_retriever", "irish_terrier", "affenpinscher", "border_collie", "hare", "boxer", "silky_terrier", "beagle", "leonberg", "haired_pointer", "patas", "dhole", "baboon", "macaque", "chesapeake_bay_retriever", "bull_mastiff", "kuvasz", "capuchin", "pug", "coated_retriever", "norwich_terrier", "coated_retriever", "hog", "keeshond", "eskimo_dog", "brittany_spaniel", "standard_poodle", "lakeland_terrier", "snow_leopard", "gordon_setter", "dingo", "standard_schnauzer", "hamster", "tibetan_terrier", "arctic_fox", "haired_fox_terrier", "basset", "water_buffalo", "american_black_bear", "angora", "bison", "howler_monkey", "hippopotamus", "chow", "giant_panda", "american_staffordshire_terrier", "shetland_sheepdog", "great_pyrenees", "chihuahua", "tabby", "marmoset", "labrador_retriever", "saint_bernard", "armadillo", "samoyed", "bluetick", "redbone", "polecat", "marmot", "kelpie", "gibbon", "llama", "miniature_pinscher", "wood_rabbit", "italian_greyhound", "lion", "cocker_spaniel", "irish_setter", "dugong", "indian_elephant", "beaver", "sussex_spaniel", "pembroke", "blenheim_spaniel", "madagascar_cat", "rhodesian_ridgeback", "lynx", "african_hunting_dog", "langur", "ibizan_hound", "timber_wolf", "cheetah", "english_foxhound", "briard", "sloth_bear", "border_terrier", "german_shepherd", "otter", "koala", "tusker", "echidna", "wallaby", "platypus", "wombat", "revolver", "umbrella", "schooner", "soccer_ball", "accordion", "ant", "starfish", "chambered_nautilus", "grand_piano", "laptop", "strawberry", "airliner", "warplane", "airship", "balloon", "space_shuttle", "fireboat", "gondola", "speedboat", "lifeboat", "canoe", "yawl", "catamaran", "trimaran", "container_ship", "liner", "pirate", "aircraft_carrier", "submarine", "wreck", "half_track", "tank", "missile", "bobsled", "dogsled", "two", "mountain_bike", "freight_car", "passenger_car", "barrow", "shopping_cart", "motor_scooter", "forklift", "electric_locomotive", "steam_locomotive", "amphibian", "ambulance", "beach_wagon", "cab", "convertible", "jeep", "limousine", "minivan", "model_t", "racer", "sports_car", "kart", "golfcart", "moped", "snowplow", "fire_engine", "garbage_truck", "pickup", "tow_truck", "trailer_truck", "moving_van", "police_van", "recreational_vehicle", "streetcar", "snowmobile", "tractor", "mobile_home", "tricycle", "unicycle", "horse_cart", "jinrikisha", "oxcart", "bassinet", "cradle", "crib", "poster", "bookcase", "china_cabinet", "medicine_chest", "chiffonier", "table_lamp", "file", "park_bench", "barber_chair", "throne", "folding_chair", "rocking_chair", "studio_couch", "toilet_seat", "desk", "pool_table", "dining_table", "entertainment_center", "wardrobe", "granny_smith", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard_apple", "pomegranate", "acorn", "hip", "ear", "rapeseed", "corn", "buckeye", "organ", "upright", "chime", "drum", "gong", "maraca", "marimba", "steel_drum", "banjo", "cello", "violin", "harp", "acoustic_guitar", "electric_guitar", "cornet", "french_horn", "trombone", "harmonica", "ocarina", "panpipe", "bassoon", "oboe", "sax", "flute", "daisy", "s_slipper", "cliff", "valley", "alp", "volcano", "promontory", "sandbar", "coral_reef", "lakeside", "seashore", "geyser", "hatchet", "cleaver", "letter_opener", "plane", "power_drill", "lawn_mower", "hammer", "corkscrew", "can_opener", "plunger", "screwdriver", "shovel", "plow", "chain_saw", "cock", "hen", "ostrich", "brambling", "goldfinch", "house_finch", "junco", "indigo_bunting", "robin", "bulbul", "jay", "magpie", "chickadee", "water_ouzel", "kite", "bald_eagle", "vulture", "great_grey_owl", "black_grouse", "ptarmigan", "ruffed_grouse", "prairie_chicken", "peacock", "quail", "partridge", "african_grey", "macaw", "crested_cockatoo", "lorikeet", "coucal", "bee_eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake", "breasted_merganser", "goose", "black_swan", "white_stork", "black_stork", "spoonbill", "flamingo", "american_egret", "little_blue_heron", "bittern", "crane", "limpkin", "american_coot", "bustard", "ruddy_turnstone", "backed_sandpiper", "redshank", "dowitcher", "oystercatcher", "european_gallinule", "pelican", "king_penguin", "albatross", "great_white_shark", "tiger_shark", "hammerhead", "electric_ray", "stingray", "barracouta", "coho", "tench", "goldfish", "eel", "rock_beauty", "anemone_fish", "lionfish", "puffer", "sturgeon", "gar", "loggerhead", "leatherback_turtle", "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana", "american_chameleon", "whiptail", "agama", "frilled_lizard", "alligator_lizard", "gila_monster", "green_lizard", "african_chameleon", "komodo_dragon", "triceratops", "african_crocodile", "american_alligator", "thunder_snake", "ringneck_snake", "hognose_snake", "green_snake", "king_snake", "garter_snake", "water_snake", "vine_snake", "night_snake", "boa_constrictor", "rock_python", "indian_cobra", "green_mamba", "sea_snake", "horned_viper", "diamondback", "sidewinder", "european_fire_salamander", "common_newt", "eft", "spotted_salamander", "axolotl", "bullfrog", "tree_frog", "tailed_frog", "whistle", "wing", "paintbrush", "hand_blower", "oxygen_mask", "snorkel", "loudspeaker", "microphone", "screen", "mouse", "electric_fan", "oil_filter", "strainer", "space_heater", "stove", "guillotine", "barometer", "rule", "odometer", "scale", "analog_clock", "digital_clock", "wall_clock", "hourglass", "sundial", "parking_meter", "stopwatch", "digital_watch", "stethoscope", "syringe", "magnetic_compass", "binoculars", "projector", "sunglasses", "loupe", "radio_telescope", "bow", "cannon", "assault_rifle", "rifle", "projectile", "computer_keyboard", "typewriter_keyboard", "crane", "lighter", "abacus", "cash_machine", "slide_rule", "desktop_computer", "held_computer", "notebook", "web_site", "harvester", "thresher", "printer", "slot", "vending_machine", "sewing_machine", "joystick", "switch", "hook", "car_wheel", "paddlewheel", "pinwheel", "s_wheel", "gas_pump", "carousel", "swing", "reel", "radiator", "puck", "hard_disc", "sunglass", "pick", "car_mirror", "solar_dish", "remote_control", "disk_brake", "buckle", "hair_slide", "knot", "combination_lock", "padlock", "nail", "safety_pin", "screw", "muzzle", "seat_belt", "ski", "candle", "lantern", "spotlight", "torch", "neck_brace", "pier", "tripod", "maypole", "mousetrap", "spider_web", "trilobite", "harvestman", "scorpion", "black_and_gold_garden_spider", "barn_spider", "garden_spider", "black_widow", "tarantula", "wolf_spider", "tick", "centipede", "isopod", "dungeness_crab", "rock_crab", "fiddler_crab", "king_crab", "american_lobster", "spiny_lobster", "crayfish", "hermit_crab", "tiger_beetle", "ladybug", "ground_beetle", "horned_beetle", "leaf_beetle", "dung_beetle", "rhinoceros_beetle", "weevil", "fly", "bee", "grasshopper", "cricket", "walking_stick", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "admiral", "ringlet", "monarch", "cabbage_butterfly", "sulphur_butterfly", "lycaenid", "jellyfish", "sea_anemone", "brain_coral", "flatworm", "nematode", "conch", "snail", "slug", "sea_slug", "chiton", "sea_urchin", "sea_cucumber", "iron", "espresso_maker", "microwave", "dutch_oven", "rotisserie", "toaster", "waffle_iron", "vacuum", "dishwasher", "refrigerator", "washer", "crock_pot", "frying_pan", "wok", "caldron", "coffeepot", "teapot", "spatula", "altar", "triumphal_arch", "patio", "steel_arch_bridge", "suspension_bridge", "viaduct", "barn", "greenhouse", "palace", "monastery", "library", "apiary", "boathouse", "church", "mosque", "stupa", "planetarium", "restaurant", "cinema", "home_theater", "lumbermill", "coil", "obelisk", "totem_pole", "castle", "prison", "grocery_store", "bakery", "barbershop", "bookshop", "butcher_shop", "confectionery", "shoe_shop", "tobacco_shop", "toyshop", "fountain", "cliff_dwelling", "yurt", "dock", "brass", "megalith", "bannister", "breakwater", "dam", "chainlink_fence", "picket_fence", "worm_fence", "stone_wall", "grille", "sliding_door", "turnstile", "mountain_tent", "scoreboard", "honeycomb", "plate_rack", "pedestal", "beacon", "mashed_potato", "bell_pepper", "head_cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti_squash", "acorn_squash", "butternut_squash", "cucumber", "artichoke", "cardoon", "mushroom", "shower_curtain", "jean", "carton", "handkerchief", "sandal", "ashcan", "safe", "plate", "necklace", "croquet_ball", "fur_coat", "thimble", "pajama", "running_shoe", "cocktail_shaker", "chest", "manhole_cover", "modem", "tub", "tray", "balance_beam", "bagel", "prayer_rug", "kimono", "hot_pot", "whiskey_jug", "knee_pad", "book_jacket", "spindle", "ski_mask", "beer_bottle", "crash_helmet", "bottlecap", "tile_roof", "mask", "maillot", "petri_dish", "football_helmet", "bathing_cap", "teddy", "holster", "pop_bottle", "photocopier", "vestment", "crossword_puzzle", "golf_ball", "trifle", "suit", "water_tower", "feather_boa", "cloak", "red_wine", "drumstick", "shield", "christmas_stocking", "hoopskirt", "menu", "stage", "bonnet", "meat_loaf", "baseball", "face_powder", "scabbard", "sunscreen", "beer_glass", "woods", "guacamole", "lampshade", "wool", "hay", "bow_tie", "mailbag", "water_jug", "bucket", "dishrag", "soup_bowl", "eggnog", "mortar", "trench_coat", "paddle", "chain", "swab", "mixing_bowl", "potpie", "wine_bottle", "shoji", "bulletproof_vest", "drilling_platform", "binder", "cardigan", "sweatshirt", "pot", "birdhouse", "hamper", "pong_ball", "pencil_box", "phone", "consomme", "apron", "punching_bag", "backpack", "groom", "bearskin", "pencil_sharpener", "broom", "mosquito_net", "abaya", "mortarboard", "poncho", "crutch", "polaroid_camera", "space_bar", "cup", "racket", "traffic_light", "quill", "radio", "dough", "cuirass", "military_uniform", "lipstick", "shower_cap", "monitor", "oscilloscope", "mitten", "brassiere", "french_loaf", "vase", "milk_can", "rugby_ball", "paper_towel", "earthstar", "envelope", "miniskirt", "cowboy_hat", "trolleybus", "perfume", "bathtub", "hotdog", "coral_fungus", "bullet_train", "pillow", "toilet_tissue", "cassette", "s_kit", "ladle", "stinkhorn", "lotion", "hair_spray", "academic_gown", "dome", "crate", "wig", "burrito", "pill_bottle", "chain_mail", "theater_curtain", "window_shade", "barrel", "washbasin", "ballpoint", "basketball", "bath_towel", "cowboy_boot", "gown", "window_screen", "agaric", "cellular_telephone", "nipple", "barbell", "mailbox", "lab_coat", "fire_screen", "minibus", "packet", "maze", "pole", "horizontal_bar", "sombrero", "pickelhaube", "rain_barrel", "wallet", "cassette_player", "comic_book", "piggy_bank", "street_sign", "bell_cote", "fountain_pen", "windsor_tie", "volleyball", "overskirt", "sarong", "purse", "bolo_tie", "bib", "parachute", "sleeping_bag", "television", "swimming_trunks", "measuring_cup", "espresso", "pizza", "breastplate", "shopping_basket", "wooden_spoon", "saltshaker", "chocolate_sauce", "ballplayer", "goblet", "gyromitra", "stretcher", "water_bottle", "dial_telephone", "soap_dispenser", "jersey", "school_bus", "jigsaw_puzzle", "plastic_bag", "reflex_camera", "diaper", "band_aid", "ice_lolly", "velvet", "tennis_ball", "gasmask", "doormat", "loafer", "ice_cream", "pretzel", "quilt", "maillot", "tape_player", "clog", "ipod", "bolete", "scuba_diver", "pitcher", "matchstick", "bikini", "sock", "cd_player", "lens_cap", "thatch", "vault", "beaker", "bubble", "cheeseburger", "parallel_bars", "flagpole", "coffee_mug", "rubber_eraser", "stole", "carbonara", "dumbbell",]
    return [str(i) for i in range(1, get_num_labels(name))]

def get_normalization_shape(name):
    if name.startswith("cifar"):
        return (3, 1, 1)
    if name == "imagenet":
        return (3, 1, 1)
    if name == "ds-imagenet":
        return (3, 1, 1)
    if name == "svhn":
        return (3, 1, 1)
    if name == "mnist":
        return (1, 1, 1)
    if name == "fashion":
        return (1, 1, 1)

def get_normalization_stats(name):
    if name == "cifar" or name == "cifar100":
        return {"mu": [0.4914, 0.4822, 0.4465], "sigma": [0.2023, 0.1994, 0.2010]}
    if name == "imagenet" or name == "ds-imagenet":
        return {"mu": [0.485, 0.456, 0.406], "sigma": [0.229, 0.224, 0.225]}
    if name == "svhn":
        return {"mu": [0.436, 0.442, 0.471], "sigma": [0.197, 0.200, 0.196]}
    if name == "mnist":
        return {"mu": [0.1307,], "sigma": [0.3081,]}
    if name == "fashion":
        return {"mu": [0.2849,], "sigma": [0.3516,]}

def get_dataset(name, split, precision):

    precision_transform = PrecisionTransform(precision)

    if name == "cifar" and split == "train":
        return datasets.CIFAR10("./data/cifar_10", train=True, download=True,
                                transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),
                                                              precision_transform]))

    if name == "cifar" and split == "test":
        return datasets.CIFAR10("./data/cifar_10", train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              precision_transform]))

    if name == "cifar" and split == "train_train":
        data = datasets.CIFAR10("./data/cifar_10", train=True, download=True,
                                transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),
                                                              precision_transform]))
        train_idxs, val_idxs = get_train_val_split(np.array(data.targets), 1000)
        return Subset(data, train_idxs)

    if name == "cifar" and split == "train_val":
        data = datasets.CIFAR10("./data/cifar_10", train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              precision_transform]))
        train_idxs, val_idxs = get_train_val_split(np.array(data.targets), 1000)
        return Subset(data, val_idxs)

    if name == "cifar100" and split == "train":
        return datasets.CIFAR100("./data/cifar_100", train=True, download=True,
                                 transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               precision_transform]))

    if name == "cifar100" and split == "test":
        return datasets.CIFAR100("./data/cifar_100", train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               precision_transform]))

    if name == "cifar100" and split == "train_train":
        data = datasets.CIFAR100("./data/cifar_100", train=True, download=True,
                                 transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               precision_transform]))
        train_idxs, val_idxs = get_train_val_split(np.array(data.targets), 1000)
        return Subset(data, train_idxs)

    if name == "cifar100" and split == "train_val":
        data = datasets.CIFAR100("./data/cifar_100", train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               precision_transform]))
        train_idxs, val_idxs = get_train_val_split(np.array(data.targets), 1000)
        return Subset(data, val_idxs)

    if name.startswith("cifar100c") and split == "test":
        _, corruption_name, corruption_severity = name.split("-")
        return CIFAR100CDataset(f"/mnt/vlgrounding/cifar_100_c/{corruption_name}.npy",
                                f"/mnt/vlgrounding/cifar_100_c/labels.npy",
                                int(corruption_severity),
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              precision_transform]))

    if name == "imagenet" and split == "train":
        return ZipData("/mnt/imagenet/train.zip", "/mnt/imagenet/train_map.txt",
                       transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           precision_transform]))

    if name == "imagenet" and split == "test":
        return ZipData("/mnt/imagenet/val.zip", "/mnt/imagenet/val_map.txt",
                       transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           precision_transform]))

    if name == "ds-imagenet" and split == "train":
        return DownsampledImageNet("/mnt/vlgrounding/downsampled_imagenet", "train",
                                   transforms.Compose([transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       precision_transform]))

    if name == "ds-imagenet" and split == "test":
        return DownsampledImageNet("/mnt/vlgrounding/downsampled_imagenet", "test",
                                   transforms.Compose([transforms.ToTensor(),
                                                       precision_transform]))

    if name == "mnist":
        return datasets.MNIST("./data/mnist", train=(split == "train"), download=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            precision_transform]))
    if name == "fashion":
        return datasets.FashionMNIST("./data/fashion", train=(split == "train"), download=True,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   precision_transform]))

    if name == "svhn":
        return datasets.SVHN("./data/svhn", split=split, download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           precision_transform]))

    raise ValueError


class DownsampledImageNet(Dataset):
    """
    Downsampled ILSVRC dataset with 1000 labels and ~1.3M images, to size 32x32.

    Source: Chrabaszcz et al. 2017.
    Notes: requires access to ImageNet to download.
    """
    train_batches = [f"train_data_batch_{i}" for i in range(1, 10 + 1)]
    test_batches = ["val_data"]

    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.batch_list = self.train_batches if split == "train" else self.test_batches

        self.data, self.labels = [], []
        for fentry in self.batch_list:
            with open(f"{self.root}/{fentry}", 'rb') as fo:
                entry = pickle.load(fo)
                self.data.append(entry["data"])
                self.labels += [label - 1 for label in entry["labels"]]

        # convert to (batch, height, width, channel) for PIL
        self.data = np.concatenate(self.data)
        self.data = self.data.reshape((self.data.shape[0], 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        img = self.transform(img) if self.transform is not None else img
        target = self.target_transform(target) if self.target_transform is not None else target
        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10SelfTrained(Dataset):
    """
    Self-trained labels for a 500K subset of 80 Million Tiny Images corresponding to CIFAR-10.

    Source: Carmon et al. NeurIPS 2019.
    """
    def __init__(self, path, transform=None, target_transform=None):
        with open(path, "rb") as fd:
            self.dataset = pickle.load(fd)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.dataset["data"][index], self.dataset["extrapolated_targets"][index]
        img = Image.fromarray(img)
        img = self.transform(img) if self.transform is not None else img
        target = self.target_transform(target) if self.target_transform is not None else target
        return img, target

    def __len__(self):
        return len(self.dataset["extrapolated_targets"])


class CIFAR100CDataset(Dataset):
    """
    Test split only
    """
    def __init__(self, x_path, y_path, severity=1, transform=None, target_transform=None):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        self.x = self.x[start_idx:end_idx]
        self.y = self.y[start_idx:end_idx]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.x[index], self.y[index]
        img = Image.fromarray(img)
        img = self.transform(img) if self.transform is not None else img
        target = self.target_transform(target) if self.target_transform is not None else target
        return img, target

    def __len__(self):
        return len(self.x)

