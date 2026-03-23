import os
import subprocess
import concurrent.futures

# 1. Read your perfect list of 100,147 links
with open('/home/azwad/Works/Multimodal-CBM/Preprocessing/physionet.org/files/gcs_cxr_urls.txt', 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

PROJECT_ID = "mimic-download-490512"

def download_folder(gs_url):
    """This worker creates the exact folder and downloads the image into it."""
    
    # Extract the true path: gs://.../files/p17/p17195991/s55012421/ -> p17/p17195991/s55012421/
    folder_path = gs_url.split('/files/')[-1]
    
    # We will save everything inside a clean master folder
    local_dir = os.path.join('mimic_cxr_organized', folder_path)
    
    # Create the exact nested folders on your hard drive
    os.makedirs(local_dir, exist_ok=True)
    
    # Tell Google Cloud to safely download the image directly into this new folder
    cmd = [
        'gsutil', '-q', '-u', PROJECT_ID, 
        'cp', '-n', f"{gs_url}*", local_dir
    ]
    subprocess.run(cmd)
    
    print(f"Safely organized: {folder_path}")

print(f"Starting highly-organized parallel download for {len(urls)} patient folders...")

# 2. Launch 20 parallel workers to do this at high speed
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(download_folder, urls)

print("All X-rays successfully downloaded and perfectly organized!")