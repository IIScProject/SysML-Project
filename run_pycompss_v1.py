from argparse import ArgumentParser

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser(description='Sys ML Project')
    parser.add_argument(
        '--model', type=str, default='rnn_pytorch', choices=
        ['rnn_pytorch', 'rnn_pytorch_stack', 'rnn_pytorch_standard','lstm_pytorch', 'lstm_pytorch_stack'],
        help='Model'
    )

    parser.add_argument(
        '--stack_length', type=int, default=1,
        help='Stack Length')

    parser.add_argument(
        '--sequence_length', type=int, default=5,
        help='Sequence Length'
    )

    parser.add_argument(
        '--num_iters', type=int, default=100,
        help='Number of iterations for training')

    parser.add_argument(
        '--batch_size', type=int, default=512,
        help='Batch size for training')


    parser.add_argument(
        '--lr', type=float, default=0.01,
        help='Learning rate for training')

    parser.add_argument(
        '--checkpoint_path', type=str,
        help='Checkpoint path')

    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device')


    parser.add_argument(
        '--l2_lambda', type=float, default=0.1,
        help='L2 regularization for training')

    parser.add_argument(
        '--get_params', type=bool, default=False,
        help='Get Parameters for model')

    args = parser.parse_args()
    if args.model == 'rnn_pytorch' or args.model == 'rnn_pytorch_stack' or args.model == 'rnn_pytorch_standard':
        if not args.get_params :
            from models.rnn_pytorch_training import main
        else :
            from models.rnn_pytorch_params import main
    else :
        exit("Invalid Model call")

    # Run the main function
    output = main(args)