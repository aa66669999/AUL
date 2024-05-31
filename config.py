import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10,
                        help="Seed for the code")
    parser.add_argument('--data_name', type=str, default="Delicious",  # Delicious bibTex
                        help="Name of Dataset")
    # parser.add_argument('--data_name', type=str, default="bibTex",
    #                     help="Name of Dataset")
    parser.add_argument('--train', type=float, default=0.005,  # 0.05
                        help="train size")  # 0.01
    parser.add_argument('--pool', type=float, default=0.7,  # 0.75
                        help="pool size")
    parser.add_argument('--test', type=float, default=0.2,
                        help="test size")

    # model settings
    parser.add_argument('--m_hidden', type=int, default=512, help="Number of nodes per hidden layer")
    parser.add_argument('--m_embed', type=int, default=512, help="Number of features in embedding space")
    parser.add_argument('--m_activation', type=str, default="ELU", help="Type of activation function used in model")
    parser.add_argument('--m_drop_p', type=float, default=0.1, help="Dropout ratio")

    # dataloader setting
    # parser.add_argument('--batch_size', type=int, default=64, help="Seed for the code")
    parser.add_argument('--batch_size', type=int, default=32, help="Seed for the code")

    # optimizer settings
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")  # 1e-4
    parser.add_argument('--wd', type=float, default=0, help="Weight decay")
    parser.add_argument('--pretrain_epochs', type=int, default=60, help="Number of weights pretraining epochs")

    # active learning setting
    parser.add_argument('--active_rounds', type=int, default=20, help="rounds of active learning")
    parser.add_argument('--active_instances', type=int, default=30, help="Number of instances for each active "
                                                                          "learning round")
    parser.add_argument('--active_epochs', type=int, default=80, help="number of active learning epochs")
    parser.add_argument('--active_batch_size', type=int, default=40, help="number of active learning epochs")

    args = parser.parse_args()
    return args
