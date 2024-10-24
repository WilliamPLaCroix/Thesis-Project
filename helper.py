"""
:)
"""
from itertools import product
import argparse

def finetune_adapters():
    import lora_finetune
    model_grades = [2, 4, 6, 8, 10, 12]
    for grade in model_grades:
        print("#"*50)
        print(f"LoRA run {grade}")
        lora_finetune.main(grade)
        print("#"*50)
        print("Training complete")
        print("#"*50)

def evaluate():
    import lora_eval
    #model_grades = {-1, 0, 1, 2, 4, 6, 8, 10, 12}
    model_grades = {0, 1, 2, 4, 6, 8, 10, 12}
    test_set_grades = {3, 5, 7, 9, 11}
    
    model_test_combos = product(model_grades, test_set_grades)
    runs = len(model_grades) * len(test_set_grades)

    for i, (model_grade, test_set_grade) in enumerate(model_test_combos):
        print("#"*50)
        print(f"LoRA run {i+1}/{runs}")
        lora_eval.main(model_grade, test_set_grade)
        print("#"*50)
        print("Evaluation complete")
        print("#"*50)
    
def merge_eval():
    import lora_merge_eval
    test_set_grades = {3, 5, 7, 9, 11}
    mixing_proportions = {1, 3, 5, 7, 9}
    model_test_combos = product(test_set_grades, mixing_proportions)
    runs = len(test_set_grades) * len(mixing_proportions)
    for i, (test_set_grades, mixing_proportions) in enumerate(model_test_combos):
        print("#"*50)
        print(f"Merge LoRA run {i+1}/{runs}")
        lora_merge_eval.main(test_set_grades, mixing_proportions)
        print("#"*50)
        print("Evaluation complete")
        print("#"*50)

def pretrain_baseline():
    import lora_baseline_adapter
    for mode in ["evens"]:
        print("#"*50)
        print(f"Training baseline {mode}")
        lora_baseline_adapter.main(mode)
        print("#"*50)
        print("Training complete")
        print("#"*50)

def main():
    parser = argparse.ArgumentParser(
                    prog='Text simplification helper script',
                    description='Helper script for training and evaluating text simplification models',
                    epilog='Enjoy training! :)')
    parser.add_argument('-m', '--mode',
                        type=str,
                        help='Must be "b", "t", "ta", "e", or "em"',
                        dest='mode',
                        required=True,
                        )
    args = parser.parse_args()


    if args.mode == "t":
        finetune_adapters()
    elif args.mode == "e":
        evaluate()
    elif args.mode == "em":
        merge_eval()
    elif args.mode == "ea":
        evaluate()
        merge_eval()
    elif args.mode == "b":
        pretrain_baseline()
    elif args.mode == "ta":
        pretrain_baseline()
        finetune_adapters()
    else:
        print('Invalid mode. Must be "b", "t", "ta", "e", or "em"')

if __name__ == "__main__":
    main()
