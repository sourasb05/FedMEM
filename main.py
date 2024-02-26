from src.Fedavg.FedAvgServer import FedAvg
#from src.PerMFL.PerMFLServer import PerMFL
from src.Fedmem.FedMEMServer import Fedmem
from src.FedProx.FedProxServer import FedProx
#from src.FeSEM.FeSEM_server import FeSEM
#from src.DemLearn.FLAlgorithms.servers.serverDemLearn import DemLearn
# from src.Optimizer.Optimizer import PerMFL
from src.TrainModels.trainmodels import *
# from src.utils.data_process import read_data
from src.utils.options import args_parser
import torchvision.models as models
import torch
from tqdm import tqdm, trange
import os


torch.manual_seed(0)


def main(args):

    # print torch.device()
    # Get device status: Check GPU or CPU

    # pbar = tqdm(times=args.times)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    current_directory = os.getcwd()
    print(current_directory)
    i = args.exp_start
    while i < args.times:
        try:
            if (args.model_name == "SimpleCNN"):
                model = SimpleCNN().to(device)
            elif(args.model_name == "AMemNetModel"):
                model = AMemNetModel().to(device)
            else:
                model = ResNet50TL().to(device)
        except ValueError:
            raise ValueError("Wrong model selected")
        try:    
            if args.algorithm == "FedAvg":
                server = FedAvg(device, model, args,i, current_directory)
            elif args.algorithm == "FedProx":
                server = FedProx(device, model, args,i, current_directory)
            elif args.algorithm == "PerMFL":
                server = PerMFL(device, model, args, i, current_directory)
            elif args.algorithm == "Fedmem":
                server = Fedmem(device, model, args, i, current_directory)
            elif args.algorithm == "FeSEM":
                server = FeSEM(device, args, i, current_directory)
            # elif args.algorithm == "DemLearn":
            #    server = DemLearn(device, args, i , current_directory)
            
        except ValueError:
            raise ValueError("Wrong algorithm selected")
        server.train()
        i+=1
        # server.test()
    # pbar.close()
if __name__ == "__main__":
    args = args_parser()
    
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("alpha       : {}".format(args.alpha))
    print("beta        : {}".format(args.beta))
    print("gamma       : {}".format(args.gamma))
    print("lamda       : {}".format(args.lamda))
    print("number of teams : {}".format(args.num_teams))
    print("eta         : {}".format(args.eta))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of team rounds          : {}".format(args.num_team_iters))
    print("Number of local rounds       : {}".format(args.local_iters))
    print("Local Model       : {}".format(args.model_name))
    print("=" * 80)

    
    main(args)