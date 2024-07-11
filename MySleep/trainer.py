import warnings
warnings.filterwarnings('ignore')
import argparse
import importlib
import os
from train import train,train_NSA

def run(args, db, gpu, from_fold, to_fold, suffix='', random_seed=42,prime=8):
    # Set GPU visible

    # Config file
    config_file = os.path.join('config', f'{db}.py')
    # config_file = "/home/denglongyan/papercode/MySleep/config/mass.py"
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Output directory
    output_dir = f'out_{db}{suffix}'

    assert from_fold <= to_fold
    assert to_fold < config.params['n_folds']

    # Training
    for fold_idx in range(from_fold, to_fold+1):
    # for fold_idx in range(from_fold, 4+1):
        train(
            args=args,
            config_file=config_file,
            fold_idx=fold_idx%prime,
            # fold_idx=fold_idx,
            output_dir=os.path.join(output_dir, 'train'),
            log_file=os.path.join(output_dir, f'Ablation.log'),
            restart=True,
            random_seed=random_seed+fold_idx,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str,  default="sleepedf")
    parser.add_argument("--gpu", type=int,  default=0)
    parser.add_argument("--from_fold", type=int,  default=0)
    parser.add_argument("--to_fold", type=int,  default=19)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--test_seq_len", type=int, default=20)
    parser.add_argument("--test_batch_size", type=int, default=15)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--prime", type=int, default=8)
    args = parser.parse_args()

    run(
        args=args,
        db=args.db,
        gpu=args.gpu,
        from_fold=args.from_fold,
        to_fold=args.to_fold,
        suffix=args.suffix,
        random_seed=args.random_seed,
        prime=args.prime,
    )
