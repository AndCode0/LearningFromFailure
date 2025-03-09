import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from util import IdxDataset, MultiDimAverageMeter, set_seed, print_dict_as_table

def parse_args():
    parser = argparse.ArgumentParser(description='Test LfF model on test set')
    parser.add_argument('-wp','--weights_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('-d','--data_dir', type=str, default="C:\\Users\\aconte\\Desktop",
                        help='Root directory for dataset storage')
    parser.add_argument('-o','--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('-s','--single_model', action='store_true',
                        help='Test a single ResNet model instead of LfF model')
    args = parser.parse_args()
    
    match args.weights_path.split('\\')[:-1]:
        case ["Models","Weights",("CelebA-ResNet" | "ColoredMNIST-SCN") as heads]:
            dataset, _ = heads.split('-')
            args.dataset_tag = dataset
        case _:
            raise ValueError
        
    if "CelebA" in args.dataset_tag:
        args.dataset_tag = "CelebA"
        args.target_attr_idx = 9
        args.bias_attr_idx = 20 
    else:
        raise NotImplementedError
        args.target_attr_idx = 0     
        args.bias_attr_idx = 1
        args.skew_ratio = 0.02,
        args.severity = 2,      
    
    # Set other default parameters
    args.batch_size = 128
    args.num_workers = 10
    args.seed = 42
    
    return args

def setup_data(args) -> None:
        """Setup datasets and data loaders"""
        if args.dataset_tag == "CelebA":
            from Data.CelebA import CustomCelebA
            transform_test = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            test_dataset = CustomCelebA(
                root=args.data_dir,
                split="test",
                target_type="attr",
                transform=transform_test,
            ) 
            test_dataset = IdxDataset(test_dataset)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            return test_loader

        elif args.dataset_tag == "ColoredMNIST":
            from Data.c_MNIST import ColoredMNIST
            test_dataset = ColoredMNIST(args.data_dir, 'test', args.skew_ratio, args.severity)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            test_dataset = IdxDataset(test_loader)
    
            return test_loader

        else:
            raise NotImplementedError

def load_LfF(args, device):
    """Load pretrained models from checkpoint"""
    
    checkpoint = torch.load(args.weights_path, map_location=device)
    if args.dataset_tag == "CelebA":
        from torchvision.models import resnet18
        if args.single_model:
            # For single model testing
            model = resnet18(weights=None, num_classes=2).to(device)
            model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
            return model, None, device
        
        model_b = resnet18(weights=None, num_classes=2).to(device)
        model_d = resnet18(weights=None, num_classes=2).to(device) 

    elif args.dataset_tag == "ColoredMNIST":
        from Models.SimpleConv import SimpleConvNet
        model_b = SimpleConvNet(num_classes=10).to(device)
        model_d = SimpleConvNet(num_classes=10).to(device)
    
    else:
        raise NotImplementedError

    model_b.load_state_dict(checkpoint['state_dict_b'])
    model_d.load_state_dict(checkpoint['state_dict_d'])
    
    return model_b, model_d, device

def evaluate(model, test_loader, args, device):# target_attr_idx, bias_attr_idx, device):
    """Evaluate model on test set"""
    model.eval()
    
    if args.dataset_tag == "CelebA":
        val_attr = 2  # Target (BlondHair) and Bias (Male) attributes
    elif args.dataset_tag == "ColoredMNIST":
        val_attr = 10
    attr_dims = [val_attr, val_attr]
    
    attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    predictions = []
    targets = []
    bias_attrs = []
    
    with torch.no_grad():
        for _, data, attr in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, args.target_attr_idx]
            bias_label = attr[:, args.bias_attr_idx]
            
            logit = model(data)
            pred = logit.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == label).float()
            
            # Track accuracy across target and bias attributes
            relevant_attrs = attr[:, [args.target_attr_idx, args.bias_attr_idx]]
            attrwise_acc_meter.add(correct.cpu(), relevant_attrs.cpu())
            
            # Store predictions and ground truth for further analysis
            predictions.extend(pred.cpu().numpy())
            targets.extend(label.cpu().numpy())
            bias_attrs.extend(bias_label.cpu().numpy())
    
    accs = attrwise_acc_meter.get_mean()
    
    # Calculate aligned and skewed accuracies
    eye_tsr = torch.eye(val_attr).long()  # 2 classes for binary attributes
    aligned_acc = accs[eye_tsr == 1].mean().item() * 100
    skewed_acc = accs[eye_tsr == 0].mean().item() * 100
    overall_acc = accs.mean().item() * 100
    
    # Calculate per-group accuracies for detailed reporting
    group_accs = None
    if args.dataset_tag == 'CelebA':
        aligned_acc, skewed_acc = skewed_acc, aligned_acc
        group_accs = {
            "Non-Blonde Women": accs[0, 0].item() * 100,  # BlondHair=0, Male=0
            "Blonde Women": accs[1, 0].item() * 100,      # BlondHair=1, Male=0
            "Non-Blonde Men": accs[0, 1].item() * 100,    # BlondHair=0, Male=1
            "Blonde Men": accs[1, 1].item() * 100,        # BlondHair=1, Male=1
        }
    
    results = {
        "overall_acc": overall_acc,
        "aligned_acc": aligned_acc,
        "skewed_acc": skewed_acc,
        "group_accs": group_accs,
        "attrwise_accs": accs,
        "predictions": predictions,
        "targets": targets,
        "bias_attrs": bias_attrs
    }
    
    return results


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loader = setup_data(args)

    if args.single_model:
        model, _, device = load_LfF(args, device)
        
        print("Evaluating single model...")
        results = evaluate(model, test_loader, args, device)
        
        print("\n===== Results Summary =====")
        print(f"Single Model:")
        print(f"\tOverall: {results['overall_acc']:.2f}%")
        print(f"\tAligned: {results['aligned_acc']:.2f}%")
        print(f"\tSkewed: {results['skewed_acc']:.2f}%")

        print_dict_as_table(results["group_accs"])
    else:

        model_b, model_d, device = load_LfF(args, device)
        
        print("Evaluating biased model...")
        biased_results = evaluate(
            model_b, test_loader, args, device
        )
        
        print("Evaluating debiased model...")
        debiased_results = evaluate(
            model_d, test_loader, args, device
        )
        
        print("\n===== Results Summary =====")
        print(f"Biased Model:")
        print(f"\tOverall: {biased_results['overall_acc']:.2f}%")
        print(f"\tAligned: {biased_results['aligned_acc']:.2f}%")
        print(f"\tSkewed: {biased_results['skewed_acc']:.2f}%")
        if args.dataset_tag =='CelebA':
            print_dict_as_table(biased_results["group_accs"])  
        
        print(f"\nDebiased Model:")
        print(f"\tOverall: {debiased_results['overall_acc']:.2f}%")
        print(f"\tAligned: {debiased_results['aligned_acc']:.2f}%")
        print(f"\tSkewed: {debiased_results['skewed_acc']:.2f}%")
        if args.dataset_tag =='CelebA':
            print_dict_as_table(debiased_results["group_accs"])  

if __name__ == "__main__":
    main()