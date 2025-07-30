import argparse
import logging
import os
import pprint
import sys
import yaml

import submitit

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, help='location to save submitit logs')
parser.add_argument('--batch-launch', action='store_true', help='batch-launch several config files')
parser.add_argument('--fname', type=str, help='yaml file', default='Ijepa_Params.yaml')
parser.add_argument('--partition', type=str, help='cluster partition')
parser.add_argument('--nodes', type=int, default=1, help='num. nodes')
parser.add_argument('--tasks-per-node', type=int, default=2, help='num. procs per node (for 2 GPUs)')  # Edited for 2 GPUs
parser.add_argument('--time', type=int, default=4300, help='time in minutes')

class Trainer:
    def __init__(self, fname='Ijepa_Params.yaml', load_model=None):
        self.fname = fname
        self.load_model = load_model

    def __call__(self):
        fname = self.fname
        load_model = self.load_model
        logger.info(f'called-params {fname}')

        params = None
        with open(fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)
        resume_preempt = False if load_model is None else load_model
        from src.train import main as app_main
        app_main(args=params, resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(self.fname, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)

def launch():
    executor = submitit.AutoExecutor(folder=os.path.join(args.folder, 'job_%j'), slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu='55G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node)
    config_fnames = [args.fname]
    jobs, trainers = [], []
    with executor.batch():
        for cf in config_fnames:
            fb_trainer = Trainer(cf)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)
    for job in jobs:
        print(job.job_id)

if __name__ == '__main__':
    args = parser.parse_args()
    launch()
