from itertools import product

import lora_eval
import lora_merge_eval

def main():
    model_grades = {-1, 0, 1, 2, 4, 6, 8, 10, 12}
    test_set_grades = {1, 3, 5, 7, 9, 11}
    model_test_combos = product(model_grades, test_set_grades)

    for mode_grade, test_set_grade in model_test_combos:
        print("#"*50)
        print(f"Running evaluation on model_grade: {mode_grade}, test_set_grade: {test_set_grade}")
        print("#"*50)
        lora_eval.main(mode_grade, test_set_grade)
        print("#"*50)
        print(f"Evaluation complete")
        print("#"*50)


    test_set_grades = {3, 5, 7, 9, 11}
    mixing_proportions = {2, 4, 6, 8}
    model_test_combos = product(test_set_grades, mixing_proportions)
    for test_set_grades, mixing_proportions in model_test_combos:
        print("#"*50)
        print(f"Running merge evaluation on test_set_grade: {test_set_grade}")
        print("#"*50)
        lora_merge_eval.main(test_set_grades, mixing_proportions)
        print("#"*50)
        print(f"Evaluation complete")
        print("#"*50)

if __name__ == "__main__":
    main()