import argparse
from src.utils import open_graph


parser = argparse.ArgumentParser(description="LED task")

# What are you doing
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")

# Data/Input Paths
parser.add_argument("--data_dir", type=str, default="../../data/way_splits/")
parser.add_argument("--image_dir", type=str, default="../../data/floorplans/")
parser.add_argument("--embedding_dir", type=str, default="../../data/word_embeddings/")
parser.add_argument("--connect_dir", type=str, default="../../data/connectivity/")
parser.add_argument(
    "--geodistance_file", type=str, default="../../data/geodistance_nodes.json"
)
# Output Paths
parser.add_argument("--summary_dir", type=str, default="../../logs/tensorboard/")
parser.add_argument("--checkpoint_dir", type=str, default="../../logs/checkpoints/")
parser.add_argument("--predictions_dir", type=str, default="../../logs/predictions")
parser.add_argument("--model_save", default=False, action="store_true")
parser.add_argument(
    "--eval_ckpt",
    type=str,
    default="/path/to/ckpt.pt",
    help="a checkpoint to evaluate by either testing or generate_predictions",
)

# FO Layer before lingunet and scaling for the image
parser.add_argument("--freeze_resnet", default=True, action="store_true")
parser.add_argument("--ds_percent", type=float, default=1.)
parser.add_argument("--ds_scale", type=float, default=0.125)
parser.add_argument("--ds_height_crop", type=int, default=54)
parser.add_argument("--ds_width_crop", type=int, default=93)
parser.add_argument("--ds_height", type=int, default=448)
parser.add_argument("--ds_width", type=int, default=448)
parser.add_argument("--max_floors", type=int, default=5)

# CLIP 
parser.add_argument("--num_maps", type=int, default=5)
parser.add_argument("--output_dim", type=int, default=1)
parser.add_argument("--input_dim", type=int, default=2048)
parser.add_argument("--batchnorm", type=bool, default=True)
parser.add_argument("--lang_fusion_type", type=str, default="mult")
parser.add_argument("--bilinear", type=bool, default=True)


# Params
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--summary", default=True, action="store_true", help="tensorboard")
parser.add_argument("--run_name", type=str, default="no_name", help="name of the run")
parser.add_argument("--cuda", type=str, default=0, help="which GPU to use")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--grad_clip", type=float, default=0.5, help="gradient clipping")
parser.add_argument("--num_epoch", type=int, default=40, help="upper epoch limit")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--early_stopping", type=int, default=10)

# Get scene graphs
def collect_graphs(args):
    scan_graphs = {}
    scans = [s.strip() for s in open(args.connect_dir + "scans.txt").readlines()]
    for scan_id in scans:
        scan_graphs[scan_id] = open_graph(args.connect_dir, scan_id)
    return scan_graphs


def parse_args():
    args = parser.parse_args()
    args.scan_graphs = collect_graphs(args)
    return args
