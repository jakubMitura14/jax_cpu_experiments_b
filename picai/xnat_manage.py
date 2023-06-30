import xnat
import os
session = xnat.connect('https://lssi-xnat-test.opi.org.pl', user="jmitura", password="@bx}%IRlT/Yb,ytfriMh")
subjects=session.projects['TEST'].subjects
testXnatPath= '/workspaces/jax_cpu_experiments_b/explore/xnat'

# def download(subb):
#     print(f"subb {subb}")
#     dirr=os.makedirs(f"{testXnatPath}/{subb.label}",exist_ok=True)
#     sub.download_dir(f"{testXnatPath}/{subb.label}")

# list(map(download,subjects))
for index in range(len(subjects)):
    subb= subjects[index]
    print(f"subb {type(subb)}")
    dirr=os.makedirs(f"{testXnatPath}/{subb.label}",exist_ok=True)
    subb.download_dir(f"{testXnatPath}/{subb.label}")
session.disconnect()