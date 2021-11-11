import sys
import argparse
import torch
import random
import pytorch_lightning as pl
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from module import PolarNetModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=10, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=100, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')
    parser.add_argument('--find_learning_rate', action='store_true', help="Finding learning rate.")
    parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')

    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training.")
    parser.add_argument('-d', '--data_dir', default='data')
    parser.add_argument('-m', '--model', choices=['polar','traditional'], default='polar', help='training model: polar or traditional (default: polar)')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32], help='grid size of BEV representation (default: [480,360,32])')

    args = parser.parse_args()

    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')

    if args.detect_anomaly:
        print("Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)
    
    # windows safe
    if sys.platform in ["win32"]:
        args.worker = 0

    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)

    callbacks = []

    #callbacks += [pl.callbacks.lr_monitor.LearningRateMonitor()]

    callbacks += [pl.callbacks.ModelCheckpoint(
        verbose=True,
        save_top_k=1,
        filename='{epoch}-{valid_miou}',
        monitor='valid_miou',
        mode='max'
    )]

    use_gpu = not args.gpus == 0

    trainer = pl.Trainer(
        log_gpu_memory=False,
        fast_dev_run=args.dev,
        profiler=False,
        gpus=args.gpus,
        log_every_n_steps=1,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.TensorBoardLogger("result", name="PolarNet"),
        callbacks=callbacks
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })

    #prepare dataset
    train_pt_dataset = SemKITTI(args.data_dir + '/sequences/', imageset = 'train', return_ref = True)
    val_pt_dataset = SemKITTI(args.data_dir + '/sequences/', imageset = 'val', return_ref = True)
    if args.model == 'polar':
        train_dataset=spherical_dataset(train_pt_dataset, grid_size = args.grid_size, flip_aug = True, ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=spherical_dataset(val_pt_dataset, grid_size = args.grid_size, ignore_label = 0, fixed_volume_space = True)
    elif args.model == 'traditional':
        train_dataset=voxel_dataset(train_pt_dataset, grid_size = args.grid_size, flip_aug = True, ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=voxel_dataset(val_pt_dataset, grid_size = args.grid_size, ignore_label = 0, fixed_volume_space = True)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = args.batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers = args.worker)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = 1,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = args.worker)

   
    
    model = PolarNetModel(args)


    if args.find_learning_rate:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        suggested_lr = lr_finder.suggestion()
        print("Old learning rate: ", args.learning_rate)
        args.learning_rate = suggested_lr
        print("Suggested learning rate: ", args.learning_rate)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


   
