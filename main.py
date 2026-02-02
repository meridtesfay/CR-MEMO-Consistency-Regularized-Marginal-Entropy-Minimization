import argparse
from src.model import load_model, adapt_batchnorm
from src.tta import MEMOAdapter

def run_experiment(args):
    # 1. Load Model
    model = load_model(args.arch)
    model = adapt_batchnorm(model)
    
    # 2. Initialize your improved MEMO algorithm
    adapter = MEMOAdapter(model, lr=args.lr, kl_weight=args.kl_weight)
    
    # 3. Load Data & Run Adaptation
    # [Insert logic here to load ImageNet-A/V2 and loop through test points]
    print(f"Running {args.arch} on {args.dataset} with KL weight {args.kl_weight}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CR-MEMO Evaluation")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="imagenet-a")
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0002)
    
    args = parser.parse_args()
    run_experiment(args)