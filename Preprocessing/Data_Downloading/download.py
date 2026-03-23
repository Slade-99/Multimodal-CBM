import os

print("Starting automated targeted downloads...")

# Open the text file containing all your custom download links
with open('/home/azwad/Works/Multimodal-CBM/targeted_download_commands.txt', 'r') as file:
    commands = file.readlines()

# Loop through every single command one by one
for i, cmd in enumerate(commands):
    print(f"Downloading file {i+1} of {len(commands)}...")
    
    # os.system tells your computer's terminal to run the command silently in the background
    os.system(cmd.strip())

print("All downloads complete! Your data is ready.")