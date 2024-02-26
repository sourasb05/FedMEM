import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="ResNet50TL", choices=["SimpleCNN", "AMemNetModel", "ResNet50TL"])
    
    parser.add_argument("--algorithm", type=str, default="Fedmem",
                        choices=["PerMFL", "FedAvg", "Fedmem", "FeSEM", "FedProx"])
    parser.add_argument("--user_ids", type=list, default=['16','17','18','19','22','23','25','26','27',\
    '28','29','30','31','32','33','34','35','36','37','38','39','41','42','43','44','45','46','47','48','49','51','52','53','54','55','56','57','60','61','62'])

    parser.add_argument("--batch_size", type=int, default=124)
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Regularizer for PerMFL")
    parser.add_argument("--lamda", type=float, default=0.5, 
                        help="Regularization term lambda")
    parser.add_argument("--gamma", type=float, default=3.0, 
                        help="regularization term gamma for PerMFL and scale parameter for RBF kernel in Fedmem")
    parser.add_argument("--alpha", type=float, default=0.05, 
                        help="learning rate")
    parser.add_argument("--eta", type=float, default=0.03, 
                        help="Learning rate for Teams in PerMFL")
    
    parser.add_argument("--num_global_iters", type=int, default=50)
    parser.add_argument("--num_team_iters", type=int, default=10)
    parser.add_argument("--local_iters", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    
    parser.add_argument("--times", type=int, default=1, 
                        help="running time")
    parser.add_argument("--exp_start", type=int, default=0,
                        help="experiment start no")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    
    parser.add_argument("--users_frac", type=float, default=1.0, 
                        help="selected fraction of users available per global round")
    parser.add_argument("--total_users", type=int, default=10, 
                        help="total participants")
   
    parser.add_argument("--num_teams", type=int, default=5,
                        help="Number of teams")
    parser.add_argument("--p_teams", type=int, default=1,
                        help="number of team selected per global round")
    parser.add_argument("--group_division", type = int, default=2, help=" 0 : sequential division , 1 : random division , 2 : only one group")
   

    args = parser.parse_args()

    return args