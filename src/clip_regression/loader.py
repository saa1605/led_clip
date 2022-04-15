import numpy as np
import json
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import copy
import re
from src.clip_lingunet.led_dataset import LEDDataset
import clip 

class Loader:
    def __init__(self, args):
        self.mesh2meters = json.load(open(args.image_dir + "pix2meshDistance.json"))
        self.max_length = 0
        self.max_dialog_length = 0
        self.datasets = {}
        self.args = args

    def load_image_paths(self, data, mode):
        episode_ids, scan_names, levels, mesh_conversions, dialogs = [], [], [], [], []
        for data_obj in data:
            episode_ids.append(data_obj["episodeId"])
            scan_names.append(data_obj["scanName"])
            dialogs.append(self.add_tokens(data_obj["dialogArray"]))
            level = 0
            if mode != "test":
                level = str(data_obj["finalLocation"]["floor"])
            levels.append(level)
            mesh_conversions.append(
                self.mesh2meters[data_obj["scanName"]][str(level)]["threeMeterRadius"]
                / 3.0
            )
        return episode_ids, scan_names, levels, mesh_conversions, dialogs

    def add_tokens(self, message_arr):
        return " ".join(message_arr)

    def load_locations(self, data, mode):
        if "test" in mode:
            return [[0, 0] for _ in data], ["" for _ in data]

        x = [
            [
                data_obj["finalLocation"]["pixel_coord"][1],
                data_obj["finalLocation"]["pixel_coord"][0],
            ]
            for data_obj in data
        ]

        y = [data_obj["finalLocation"]["viewPoint"] for data_obj in data]

        return x, y

    def build_dataset(self, file):
        mode = file.split("_")[0]
        print("[{}]: Loading JSON file...".format(mode))
        data = json.load(open(self.args.data_dir + file))
        print("[{}]: Using {} samples".format(mode, len(data)))
        locations, viewPoint_location = self.load_locations(data, mode)
        (
            episode_ids,
            scan_names,
            levels,
            mesh_conversions,
            dialogs,
        ) = self.load_image_paths(data, mode)
        # texts = copy.deepcopy(dialogs)
        # texts, seq_lengths = self.build_pretrained_vocab(texts)
        texts = dialogs
        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            mode,
            self.args,
            texts,
            None, # Remove this, this is where seq_lens was
            mesh_conversions,
            locations,
            viewPoint_location,
            dialogs,
            scan_names,
            levels,
            episode_ids,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))



