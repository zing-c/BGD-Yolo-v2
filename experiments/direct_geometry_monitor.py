"""Log CAM-zero rates without changing CAM, losses, optimizer, or best selection."""

import json


def reset_cam_epoch(trainer):
    trainer.model.direct_cam_counts = {}
    if trainer.ema:
        trainer.ema.ema.direct_cam_counts = {}


def log_cam_epoch(trainer):
    if getattr(trainer, '_last_cam_logged_epoch', None) == trainer.epoch:
        return  # final standalone best validation also invokes fit_epoch_end
    trainer._last_cam_logged_epoch = trainer.epoch
    val_model = trainer.ema.ema if trainer.ema else trainer.model
    row = {'epoch': trainer.epoch + 1,
           'train': dict(getattr(trainer.model, 'direct_cam_counts', {}).get('train', {})),
           'val': dict(getattr(val_model, 'direct_cam_counts', {}).get('val', {}))}
    for phase in ('train', 'val'):
        values = row[phase]
        denominator = values.get('candidate_images', 0)
        values['zero_cam_fraction_of_candidate_images'] = (
            values.get('zero_cam_candidate_images', 0) / denominator if denominator else None)
    with (trainer.save_dir / 'cam_values.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    print('CAM_VALUES_EPOCH ' + json.dumps(row), flush=True)
