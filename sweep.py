from main import *
import submitit
import os
import csv

class Cluster:
    def __init__(self,
                 checkpoint_dir="clusterlogsTanya/",
                 partition="learnfair",
                 array_parallelism=512,
                 num_cpus=8,
                 num_gpus=1,
                 time=4320):
        self.checkpoint_dir = os.path.join(os.getenv("HOME"),
                                           checkpoint_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.executor = submitit.SlurmExecutor(folder=self.checkpoint_dir)

        self.executor.update_parameters(time=time,
                                        num_gpus=num_gpus,
                                        array_parallelism=array_parallelism,
                                        cpus_per_task=num_cpus,
                                        partition=partition)
        self.jobs = []

    def submit(self, function, args):
        self.jobs += self.executor.map_array(function, args)

if __name__ == "__main__":
    all_args = []
    l = 0
    for family in ['glob', 'DNA_pol_B_exo', 'cytb', 'eIF6', 'glob', 'igvar-h', 'sh3']:
        for knn in [2, 3, 5]:
            for sigma in [1.0, 2.0]:
                for gamma in [1.0, 2.0]:
                    for batchsize in [-1, 64]:
                        l += 1
                        train_args = parse_args()
                        train_args.family = family
                        train_args.knn = knn
                        train_args.sigma = sigma
                        train_args.gamma = gamma
                        train_args.batchsize = batchsize
                        train_args.seed = 1
                        all_args.append(train_args)

    print("Launching one array of {} jobs...".format(len(all_args)))
    cluster = Cluster()
    cluster.submit(poincare_map, all_args)

    for job in cluster.jobs:
        print("Submitted job array: {}".format(job.job_id.split("_")[0]))
        break
