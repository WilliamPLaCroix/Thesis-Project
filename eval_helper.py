from itertools import product
from importlib import reload

import lora_eval
import lora_merge_eval

def main():
    model_grades = {-1, 0, 1, 2, 4, 6, 8, 10, 12}
    test_set_grades = {3, 5, 7, 9, 11}
    model_test_combos = product(model_grades, test_set_grades)
    runs = len(model_grades) * len(test_set_grades)

    for i, (model_grade, test_set_grade) in enumerate(model_test_combos):
        print("#"*50)
        print(f"LoRA run {i+1}/{runs}")
        lora_eval.main(model_grade, test_set_grade)
        print("#"*50)
        print(f"Evaluation complete")
        print("#"*50)


    test_set_grades = {3, 5, 7, 9, 11}
    mixing_proportions = {2, 4, 6, 8}
    model_test_combos = product(test_set_grades, mixing_proportions)
    runs = len(test_set_grades) * len(mixing_proportions)
    for i, (test_set_grades, mixing_proportions) in enumerate(model_test_combos):
        print("#"*50)
        print(f"Merge LoRA run {i+1}/{runs}")
        lora_merge_eval.main(test_set_grades, mixing_proportions)
        print("#"*50)
        print(f"Evaluation complete")
        print("#"*50)

if __name__ == "__main__":
    main()