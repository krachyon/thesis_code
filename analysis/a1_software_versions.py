import subprocess as sp
import re

interesting_packages = ['astropy(?!-)',
                        'anisocado',
                        'ScopeSim',
                        'photutils',
                        'thesis_lib',
                        'numpy(?!\\w)',
                        'matplotlib(?!-)',
                        'pandas',
                        'multiprocess',
                        'dill']
pip = sp.run(['pip', 'freeze'], capture_output=True)

res = pip.stdout.decode('utf8').split('\n')
packages = [i for i in res for j in interesting_packages if re.search(j, i)]
#packages = [re.search(r'((?<=@)(.*)(?=#))|(.*==.*)', i).group(0) for i in packages]
outstring = '\n'.join(packages)
with open('a1_software/versions.txt', 'w') as f:
    f.write(outstring)
print(outstring)    




