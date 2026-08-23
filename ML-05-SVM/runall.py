import subprocess
from pathlib import Path
 
# Define the target folder path
script_dir = Path(".")
 
# Find and execute every python script in alphabetical order
for file_path in sorted(script_dir.glob("*.py")):
    print(f"Executing: {file_path.name}")
    subprocess.run(["python", str(file_path)], check=True)