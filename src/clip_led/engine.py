from tqdm import tqdm 
from src.utils import distance_from_pixels, accuracy
import numpy as np 

def train_model(model, loader, loss_fn, optimizer, scaler, config):
    acc5m = []
    acc3m = []
    acc0m = []
    losses = []
    localization_errors = []
    for enum, data in enumerate(tqdm(loader)):
        optimizer.zero_grad()

        maps = data['maps'].float().to(config['device'])
        target_maps = data['target_maps'].float().to(config['device'])
        conversions = data['conversions'].float()
        dialogs = data['dialogs'].float().squeeze(1).to(config['device']) # The squeeze removes extra dimension in (BATCH_SIZE, 1, NUM_TOKENS)

        # Data Required to calculate Localization Accuracy (Geodesic)
        scan_names = data['scan_names']
        true_viewpoints = data['true_viewpoints']
        episode_ids = data['episode_ids']

        # with torch.autocast('cpu'):
        preds = model(maps, dialogs)
        loss = loss_fn(preds, target_maps)

        # scaler.scale(loss).backward()

        # scaler.step(optimizer)
        loss.backward()

        optimizer.step()

        # scaler.update()

        le, ep = distance_from_pixels(
            config, preds.detach().cpu(), conversions, scan_names, true_viewpoints, episode_ids, 'train', )
        losses.append(loss.item())
        acc5m.append(accuracy(le, 5))
        acc3m.append(accuracy(le, 3))
        acc0m.append(accuracy(le, 0))
        localization_errors.extend(le)
    return {
        'loss': np.mean(losses),
        'acc5m': np.mean(np.asarray(acc5m)),
        'acc3m': np.mean(np.asarray(acc3m)),
        'acc0m': np.mean(np.asarray(acc0m)),
    }

def eval_model(model, loader, loss_fn, config, mode):
    acc5m = []
    acc3m = []
    acc0m = []
    losses = []
    localization_errors = []
    for enum, data in enumerate(tqdm(loader)):
        maps = data['maps'].float().to(config['device'])
        target_maps = data['target_maps'].float().to(config['device'])
        conversions = data['conversions'].float()
        dialogs = data['dialogs'].float().squeeze(1).to(config['device']) # The squeeze removes extra dimension in (BATCH_SIZE, 1, NUM_TOKENS)

        # Data Required to calculate Localization Accuracy (Geodesic)
        scan_names = data['scan_names']
        true_viewpoints = data['true_viewpoints']
        episode_ids = data['episode_ids']

        # with torch.autocast('cpu'):
        preds = model(maps, dialogs)
        loss = loss_fn(preds, target_maps)


        le, ep = distance_from_pixels(
            config, preds.detach().cpu(), conversions, scan_names, true_viewpoints, episode_ids, mode)
        losses.append(loss.item())
        acc5m.append(accuracy(le, 5))
        acc3m.append(accuracy(le, 3))
        acc0m.append(accuracy(le, 0))
        localization_errors.extend(le)
    return {
        'loss': np.mean(losses),
        'acc5m': np.mean(np.asarray(acc5m)),
        'acc3m': np.mean(np.asarray(acc3m)),
        'acc0m': np.mean(np.asarray(acc0m)),
    }