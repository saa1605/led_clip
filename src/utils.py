import torch
import numpy as np
import json
import networkx as nx
import numpy as np
import math
import torch.nn as nn
import os


def evaluate(args, splitFile, run_name):
    split_name = splitFile.split("_")[0]
    distance_scores = []
    splitData = json.load(open(args.data_dir + splitFile))
    geodistance_nodes = json.load(open(args.geodistance_file))
    fileName = f"{run_name}_{split_name}_submission.json"
    fileName = os.path.join(args.predictions_dir, fileName)
    submission = json.load(open(fileName))
    for gt in splitData:
        gt_vp = gt["finalLocation"]["viewPoint"]
        pred_vp = submission[gt["episodeId"]]["viewpoint"]
        dist = geodistance_nodes[gt["scanName"]][gt_vp][pred_vp]
        distance_scores.append(dist)

    distance_scores = np.asarray(distance_scores)
    print(
        f"Result {split_name} -- \n LE: {np.mean(distance_scores):.4f}",
        f"Acc@0m: {sum(distance_scores <= 0) * 1.0 / len(distance_scores):.4f}",
        f"Acc@3m: {sum(distance_scores <= 3) * 1.0 / len(distance_scores):.4f}",
        f"Acc@5m: {sum(distance_scores <= 5) * 1.0 / len(distance_scores):.4f}",
        f"Acc@10m: {sum(distance_scores <= 10) * 1.0 / len(distance_scores):.4f}",
    )


def accuracy(dists, threshold=3):
    """Calculating accuracy at 3 meters by default"""
    return np.mean((torch.tensor(dists) <= threshold).int().numpy())


def accuracy_batch(dists, threshold):
    return (dists <= threshold).int().numpy().tolist()


def distance(pose1, pose2):
    """Euclidean distance between two graph poses"""
    return (
        (pose1["pose"][3] - pose2["pose"][3]) ** 2
        + (pose1["pose"][7] - pose2["pose"][7]) ** 2
        + (pose1["pose"][11] - pose2["pose"][11]) ** 2
    ) ** 0.5


def open_graph(connectDir, scan_id):
    """Build a graph from a connectivity json file"""
    infile = "%s%s_connectivity.json" % (connectDir, scan_id)
    G = nx.Graph()
    with open(infile) as f:
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(
                            item["image_id"],
                            data[j]["image_id"],
                            weight=distance(item, data[j]),
                        )
    return G


def get_geo_dist(D, n1, n2):
    return nx.dijkstra_path_length(D, n1, n2)


def snap_to_grid(geodistance_nodes, node2pix, sn, pred_coord, conversion, level):
    min_dist = math.inf
    best_node = ""
    for node in node2pix[sn].keys():
        if node2pix[sn][node][2] != int(level) or node not in geodistance_nodes:
            continue
        target_coord = [node2pix[sn][node][0][1], node2pix[sn][node][0][0]]
        dist = np.sqrt(
            (target_coord[0] - pred_coord[0]) ** 2
            + (target_coord[1] - pred_coord[1]) ** 2
        ) / (conversion)
        if dist.item() < min_dist:
            best_node = node
            min_dist = dist.item()
    return best_node


def distance_from_pixels(args, preds, mesh_conversions, info_elem, mode):
    """Calculate distances between model predictions and targets within a batch.
    Takes the propablity map over the pixels and returns the geodesic distance"""
    node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
    geodistance_nodes = json.load(open(args.geodistance_file))
    distances, episode_predictions = [], []
    dialogs, levels, scan_names, episode_ids, true_viewpoints = info_elem
    for pred, conversion, sn, tv, id in zip(
        preds, mesh_conversions, scan_names, true_viewpoints, episode_ids
    ):
        total_floors = len(set([v[2] for k, v in node2pix[sn].items()]))
        pred = nn.functional.interpolate(
            pred.unsqueeze(1), (700, 1200), mode="bilinear"
        ).squeeze(1)[:total_floors]
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        convers = conversion.view(args.max_floors, 1, 1)[pred_coord[0].item()]
        pred_viewpoint = snap_to_grid(
            geodistance_nodes[sn],
            node2pix,
            sn,
            [pred_coord[1].item(), pred_coord[2].item()],
            convers,
            pred_coord[0].item(),
        )
        if mode != "test":
            dist = geodistance_nodes[sn][tv][pred_viewpoint]
            distances.append(dist)
        episode_predictions.append([id, pred_viewpoint])
    return distances, episode_predictions

# def annotateImageWithSegmentationData(image, annotationDict):
#     image = np.array(image)
#     print(image.shape)
#     segmentChannel = np.zeros(image.shape)
#     imageShape = image.shape

#     for r, c in image:
#         point = Point(r, c)
#         for segment in annotationDict:
#             polygon = Polygon(segment['shapes']['points'])
#             if polygon.contains(point):
#                 segmentChannel[r, c] = segment['shapes']['label']
#                 break 
#     return segmentChannel


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    accuracy = (true_positives + true_negatives) / len(y_true)
    assert not np.isnan(true_negatives)
    assert not np.isnan(false_positives)
    assert not np.isnan(false_negatives)
    assert not np.isnan(true_positives)

    if true_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        iou = true_positives / (true_positives + false_positives + false_negatives)
    else:
        precision = recall = f1_score = iou = float('NaN')
    return {
        "accuracy": accuracy,
        "TN": true_negatives,
        "FP": false_positives,
        "FN": false_negatives,
        "TP": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "true_mean": np.mean(y_true),
        "pred_mean": np.mean(y_pred),
    }


# def visualize_data(self, index):
#     all_maps = np.zeros((
#         self.config['max_floors'],
#         self.config["image_size"][1],
#         self.config["image_size"][2],
#         self.config["image_size"][0],
#     )
#     )
#     scan_name = self.data[index]['scanName']
#     floors = self.mesh2meters[scan_name].keys()
#     images = []
#     for enum, floor in enumerate(floors):
#         img = Image.open(f'{self.image_dir}floor_{floor}/{scan_name}_{floor}.png').convert('RGB')
#         all_maps[enum] = torch.permute(self.preprocess_visualize(img)[:3, :, :], (1, 2, 0)).cpu().numpy()

#     # create figure
#     fig = plt.figure(figsize=(20, 15))
    
#     # setting values to rows and column variables
#     rows = 3
#     columns = 2
#     # Adds a subplot at the 1st position
#     fig.add_subplot(rows, columns, 1)
    
#     # showing image
#     plt.imshow(all_maps[0])
#     plt.Circle((100, 100), 50, color='k')
#     plt.axis('off')
#     plt.title("First")
    
#     # Adds a subplot at the 2nd position
#     fig.add_subplot(rows, columns, 2)
    
#     # showing image
#     plt.imshow(all_maps[1])
#     plt.axis('off')
#     plt.title("Second")
    
#     # Adds a subplot at the 3rd position
#     fig.add_subplot(rows, columns, 3)
    
#     # showing image
#     plt.imshow(all_maps[2])
#     plt.axis('off')
#     plt.title("Third")
    
#     # Adds a subplot at the 4th position
#     fig.add_subplot(rows, columns, 4)
    
#     # showing image
#     plt.imshow(all_maps[3])
#     plt.axis('off')
#     plt.title("Fourth")

#     # Adds a subplot at the 4th position
#     fig.add_subplot(rows, columns, 5)
    
#     # showing image
#     plt.imshow(all_maps[4])
#     plt.axis('off')
#     plt.title("Fourth")