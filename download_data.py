import os

print("This script will download all files required for benchmark problems."
      " You must have 'wget' installed for the downloads to work.")

folder_name = "data"
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

fnames = ["Albedo.h5",
          "BodyEIT.h5",
          "HelmholtzTomography.h5",
          "Poisson70_L20.h5"
          "Opt.zip",
          "data_8000.zip",
          "data_9000.zip"
          ]
for fid, fname in enumerate(fnames):
    print('Downloading file {fname} ({fid+1}/{len(fnames)}):')
    url = "https://zenodo.org/record/8121672/files/" + fname
    cmd = f"wget --directory-prefix {folder_name} {url}"
    os.system(cmd)
