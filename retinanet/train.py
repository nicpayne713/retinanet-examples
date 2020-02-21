import os
import time
from statistics import mean
from math import isfinite 

import mlflow
import mlflow.pytorch

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel
from .backbones.layers import convert_fixedbn_model

from .data import DataIterator
from .dali import DaliDataIterator
from .utils import ignore_sigint, post_metrics, Profiler
from .infer import infer


def train(model, state, path, annotations, val_path, val_annotations, resize, max_size,
          jitter, batch_size, iterations, val_iterations, mixed_precision, lr, warmup,
          milestones, gamma, is_master=True, world=1, use_dali=True, verbose=True,
          metrics_url=None, logdir=None, weight_decay=0.0001, momentum=0.9):
    'Train the model on the given dataset'

    params = {'weight_decay': weight_decay,
              'momentum': momentum,
              'initial_lr': lr,
              'gamma': gamma,
              'iterations': iterations,
              'jitter': jitter,
              'batch_size': batch_size,
              }
    metrics = {}
    # Prepare model
    nn_model = model
    stride = model.stride

    model = convert_fixedbn_model(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # Setup optimizer and schedule
    optimizer = SGD(model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level = 'O2' if mixed_precision else 'O0',
                                      keep_batchnorm_fp32 = True,
                                      loss_scale = 128.0,
                                      verbosity = is_master)

    if world > 1: 
        model = DistributedDataParallel(model)
    model.train()

    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    def schedule(train_iter):
        if warmup and train_iter <= warmup:
            return 0.9 * train_iter / warmup + 0.1
        return gamma ** len([m for m in milestones if m <= train_iter])
    scheduler = LambdaLR(optimizer, schedule)

    # Prepare dataset
    if verbose: print('Preparing dataset...')
    data_iterator = (DaliDataIterator if use_dali else DataIterator)(
        path, jitter, max_size, batch_size, stride,
        world, annotations, training=True)
    if verbose: print(data_iterator)


    if verbose:
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'))
        print('    batch: {}, precision: {}'.format(batch_size, 'mixed' if mixed_precision else 'full'))
        print('Training model for {} iterations...'.format(iterations))
    params['device'] = 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'

    # Create TensorBoard writer
    if logdir is not None:
        from tensorboardX import SummaryWriter
        if is_master and verbose:
            print('Writing TensorBoard logs to: {}'.format(logdir))
        writer = SummaryWriter(logdir=logdir)

    profiler = Profiler(['train', 'fw', 'bw'])
    iteration = state.get('iteration', 0)
    experiment = os.environ.get('MLFLOW_EXPERIMENT', 'retinanet-examples')
    uri = r'file:/root/app/retinanet-examples/mlruns/'
    start_time = time.time()
    timed = 0.
    if is_master:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run()
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_artifact('/images', 'images')
        mlflow.log_artifact('/summaries', 'summaries')
        mlflow.log_artifact('/models', 'models')
    while iteration < iterations:
        cls_losses, box_losses = [], []
        for i, (data, target) in enumerate(data_iterator):

            # Forward pass
            profiler.start('fw')

            optimizer.zero_grad()
            cls_loss, box_loss = model([data, target])
            del data
            profiler.stop('fw')

            # Backward pass
            profiler.start('bw')
            with amp.scale_loss(cls_loss + box_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            scheduler.step(iteration)

            # Reduce all losses
            cls_loss, box_loss = cls_loss.mean().clone(), box_loss.mean().clone()
            if world > 1:
                torch.distributed.all_reduce(cls_loss)
                torch.distributed.all_reduce(box_loss)
                cls_loss /= world
                box_loss /= world
            if is_master:
                cls_losses.append(cls_loss)
                box_losses.append(box_loss)

            if is_master and not isfinite(cls_loss + box_loss):
                raise RuntimeError('Loss is diverging!\n{}'.format(
                    'Try lowering the learning rate.'))

            del cls_loss, box_loss
            profiler.stop('bw')

            iteration += 1
            profiler.bump('train')
            if is_master and (profiler.totals['train'] > 60 or iteration == iterations):
                focal_loss = torch.stack(list(cls_losses)).mean().item()
                box_loss = torch.stack(list(box_losses)).mean().item()
                learning_rate = optimizer.param_groups[0]['lr']
                if verbose:
                    msg  = '[{:{len}}/{}]'.format(iteration, iterations, len=len(str(iterations)))
                    msg += ' focal loss: {:.3f}'.format(focal_loss)
                    msg += ', box loss: {:.3f}'.format(box_loss)
                    msg += ', {:.3f}s/{}-batch'.format(profiler.means['train'], batch_size)
                    msg += ' (fw: {:.3f}s, bw: {:.3f}s)'.format(profiler.means['fw'], profiler.means['bw'])
                    msg += ', {:.1f} im/s'.format(batch_size / profiler.means['train'])
                    msg += ', lr: {:.2g}'.format(learning_rate)
                    print(msg, flush=True)
                mlflow.log_metric('focal_loss', focal_loss)
                mlflow.log_metric('box_loss', box_loss)
                mlflow.log_metric('image/sec',
                                  batch_size / profiler.means['train'])
                mlflow.log_metric('learning_rate', learning_rate)

                # log model artifact at current iteration
                mlflow.pytorch.log_model(model, 'models')

                if logdir is not None:
                    writer.add_scalar('focal_loss', focal_loss,  iteration)
                    writer.add_scalar('box_loss', box_loss, iteration)
                    writer.add_scalar('learning_rate', learning_rate, iteration)
                    del box_loss, focal_loss

                if metrics_url:
                    post_metrics(metrics_url, {
                        'focal loss': mean(cls_losses),
                        'box loss': mean(box_losses),
                        'im_s': batch_size / profiler.means['train'],
                        'lr': learning_rate
                    })

                # Save model weights
                state.update({
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                })
                with ignore_sigint():
                    nn_model.save(state)

                profiler.reset()
                del cls_losses[:], box_losses[:]

            if val_annotations and (iteration == iterations or iteration % val_iterations == 0):
                timed = time.time() - start_time
                infer(model, val_path, None, resize, max_size, batch_size, annotations=val_annotations,
                    mixed_precision=mixed_precision, is_master=is_master, world=world,
                      use_dali=use_dali, is_validation=True, verbose=True, mlflow=mlflow)
                model.train()

            if iteration == iterations:
                if is_master:
                    if timed == 0:
                        timed = time.time() - start_time
                    mlflow.log_metric('time_to_train', timed)
                    mlflow.pytorch.log_model(model, 'models')
                    mlflow.end_run()
                break

    if logdir is not None:
        writer.close()


