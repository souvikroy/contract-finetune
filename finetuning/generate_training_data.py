#!/usr/bin/env python
"""Generate training data from contract documents."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finetuning.data_generator import TrainingDataGenerator
from src.config import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate training data from contracts')
    
    parser.add_argument('--contract_path', type=str, default=None,
                       help='Path to contract JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for training data (JSONL format)')
    parser.add_argument('--format', type=str, default='jsonl', choices=['jsonl', 'json'],
                       help='Output format')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Augment data with paraphrases')
    parser.add_argument('--num_augmentations', type=int, default=2,
                       help='Number of augmentations per example')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Use contract path from args or config
    contract_path = args.contract_path or settings.contract_json_path
    
    print(f"Generating training data from: {contract_path}")
    print(f"Output path: {args.output_path}")
    
    # Create generator
    generator = TrainingDataGenerator(contract_path=contract_path)
    
    # Generate Q&A pairs
    print("Generating Q&A pairs from contract...")
    qa_pairs = generator.generate_qa_pairs_from_contract()
    print(f"Generated {len(qa_pairs)} Q&A pairs from contract")
    
    # Generate from test cases
    print("Generating examples from test cases...")
    test_case_examples = generator.generate_from_test_cases()
    print(f"Generated {len(test_case_examples)} examples from test cases")
    
    # Combine
    all_examples = qa_pairs + test_case_examples
    
    # Format for training
    print("Formatting examples...")
    formatted = generator.format_for_training(all_examples)
    print(f"Formatted {len(formatted)} examples")
    
    # Augment if requested
    if args.augment:
        print(f"Augmenting data ({args.num_augmentations} augmentations per example)...")
        formatted = generator.augment_data(formatted, num_augmentations=args.num_augmentations)
        print(f"After augmentation: {len(formatted)} examples")
    
    # Split dataset
    print("Splitting dataset...")
    splits = generator.split_dataset(formatted, train_ratio=0.8, val_ratio=0.1)
    
    print(f"Train: {len(splits['train'])}")
    print(f"Validation: {len(splits['validation'])}")
    print(f"Test: {len(splits['test'])}")
    
    # Save datasets
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    generator.save_dataset(formatted, str(output_path), format=args.format)
    
    # Save splits
    train_path = output_path.parent / f"{output_path.stem}_train{output_path.suffix}"
    val_path = output_path.parent / f"{output_path.stem}_val{output_path.suffix}"
    test_path = output_path.parent / f"{output_path.stem}_test{output_path.suffix}"
    
    generator.save_dataset(splits['train'], str(train_path), format=args.format)
    generator.save_dataset(splits['validation'], str(val_path), format=args.format)
    generator.save_dataset(splits['test'], str(test_path), format=args.format)
    
    print(f"\nSaved datasets:")
    print(f"  Full dataset: {output_path}")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")


if __name__ == '__main__':
    main()
