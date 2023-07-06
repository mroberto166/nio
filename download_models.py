import os

print("This script will download all files required for benchmark problems."
      " You must have 'wget' installed for the downloads to work.")

folder_name = "./"

fnames = ["FinalModelNewPerm.zip"]
for fid, fname in enumerate(fnames):
    print('Downloading file {fname} ({fid+1}/{len(fnames)}):')
    url = "https://zenodo.org/record/8121672/files/" + fname
    cmd = f"wget --directory-prefix {folder_name} {url}"
    os.system(cmd)
