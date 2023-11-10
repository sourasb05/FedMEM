import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="ResNet50FC", choices=["SimpleCNN",
                                                                            "VGG16FC",
                                                                            "ResNet18FC",
                                                                            "ResNet50FC",
                                                                            "ResNet101FC",
                                                                            "ResNet50FT",
                                                                            "AMemNetModel"])
    
    parser.add_argument("--algorithm", type=str, default="FedAvg",
                        choices=["PerMFL", "FedAvg"])
    

    parser.add_argument("--batch_size", type=int, default=124)
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Regularizer for PerMFL")
    parser.add_argument("--lamda", type=float, default=0.1, 
                        help="Regularization term lambda")
    parser.add_argument("--gamma", type=float, default=3.0, 
                        help="regularization term gamma for PerMFL")
    parser.add_argument("--alpha", type=float, default=0.01, 
                        help="learning rate")
    parser.add_argument("--eta", type=float, default=0.03, 
                        help="Learning rate for Teams in PerMFL")
    
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--num_team_iters", type=int, default=10)
    parser.add_argument("--local_iters", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    
    parser.add_argument("--times", type=int, default=1, 
                        help="running time")
    parser.add_argument("--exp_start", type=int, default=0,
                        help="experiment start no")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    
    parser.add_argument("--selected_users", type=int, default=10, 
                        help="selected user per round of training")
    parser.add_argument("--numusers", type=int, default=10, 
                        help="Number of Users per round")
    parser.add_argument("--num_teams", type=int, default=1,
                        help="Number of teams")
    parser.add_argument("--p_teams", type=int, default=1,
                        help="number of team selected per global round")
    parser.add_argument("--group_division", type = int, default=2, help=" 0 : sequential division , 1 : random division , 2 : only one group")
   

    args = parser.parse_args()

    return args