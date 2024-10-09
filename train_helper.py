from importlib import reload

import lora_adapter

def main():
    model_grades = {2, 4, 6, 8, 10, 12}
    for grade in model_grades:
        print("#"*50)
        print(f"LoRA run {grade}")
        lora_adapter.main(grade)
        print("#"*50)
        print(f"Training complete")
        print("#"*50)

if __name__ == "__main__":
    main()