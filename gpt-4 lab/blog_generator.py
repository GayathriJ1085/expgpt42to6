
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

applications = {
    "1": ("Blog Generator", BASE_DIR / "blog_generator_app.py"),
    "2": ("Legal Analyzer", BASE_DIR / "exp 5" / "legal_analyzer.py"),
    "3": ("Medical Assistant", BASE_DIR / "exp 6" / "medical_assistant.py"),
    "4": ("Email Assistant", BASE_DIR / "exp-3" / "email_assistant.py"),
   "5": ("Recommendation System", BASE_DIR / "exp-4" / "recommender.py"),
}


def main():
    while True:
        print("\n================================")
        print("       AI APPLICATION HUB")
        print("================================")
        print("1. Blog Generator")
        print("2. Legal Analyzer")
        print("3. Medical Assistant")
        print("4. Email Assistant")
        print("5. Recommendation System")
        print("6. Exit")
        print("================================")

        choice = input("Enter your choice: ").strip()

        if choice == "6":
            print("Exiting AI Application Hub. Goodbye!")
            break

        if choice not in applications:
            print("Invalid choice. Please enter 1 to 6.")
            continue

        app_name, app_path = applications[choice]

        if not app_path.is_file():
            print(f"File not found: {app_path}")
            print("Check the folder name and file location.")
            continue

        print(f"\nLaunching {app_name}...\n")

        try:
            subprocess.run(
                [sys.executable, str(app_path)],
                cwd=str(app_path.parent)
            )
        except Exception as error:
            print(f"Error launching {app_name}: {error}")

        print(f"\n{app_name} finished.")


if __name__ == "__main__":
    main()
