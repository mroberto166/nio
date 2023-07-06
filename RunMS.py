import os

problems_nio = ["sine", "helm", "eit", "rad", "curve", "style"]
problems_don = ["sine", "helm", "eit", "rad", "curve", "style"]
problems_fcnn = ["sine", "helm", "eit", "rad"]

models = ["nio_new", "don", "fcnn"]
problems = [["sine", "helm", "eit", "rad", "curve", "style"],
            ["sine", "helm", "eit", "rad", "curve", "style"],
            ["sine", "helm", "eit", "rad"]]

for mod, p_list in zip(models, problems):
    for p in p_list:
        string_to_ex_1 = r'sbatch --wrap="python3 ModelSelectionNIO.py ' + str(mod) + r' ' + str(p) + r'"'
        os.system(string_to_ex_1)
