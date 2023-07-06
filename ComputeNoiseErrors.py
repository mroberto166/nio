import os

abl = False
plot = 0
if abl:
    fno = False
    if fno:
        main_folder = "FinalModelAblFNO"
    else:
        main_folder = "FinalModelAblRB"

    problems = ["sine", "helm", "eit", "rad"]
    noise = [0]

    dictionary_m = {"sine": [5, 10, 15, "false"],
                    "helm": [5, 10, 15, "false"],
                    "rad": [10, 15, 20, 25, 30, "false"],
                    "eit": [10, 15, 20, 25, 30, "false"]
                    }
else:
    main_folder = "FinalModelNewPerm"
    problems = ["sine", "eit", "helm", "rad", "curve", "style"]

    if bool(plot):
        noise = [0]
        dictionary_m = {
            "eit": ["false"]}
    else:
        noise = [0, 0.01, 0.05, 0.1]
        dictionary_m = {"sine": [5, 10, 15, "false"],
                        "helm": [5, 10, 15, "false"],
                        "helm_stab": [5, 10, 15, "false"],
                        "eit": [5, 10, 15, 20, 25, 30, "false"],
                        "curve": [1, 2, 3, 4, "false"],
                        "style": [1, 2, 3, 4, "false"],
                        "rad": [15, 20, 25, 30, "false"]}

for problem in problems:
    for n in noise:
        m_vec = dictionary_m[problem]
        print(problem, m_vec)
        for m in m_vec:
            if abl:
                string_to_exec = "sbatch --time=24:00:00 -n 1  --mem-per-cpu=24000  --wrap=\" python3 ComputeErrorsAbl.py " + str(problem) + " " + str(n) + " " + str(m) + " " + str(main_folder) + " " + str(plot) + " \""
            else:
                if problem != "rad":
                    string_to_exec = "sbatch --time=24:00:00 -n 1  --mem-per-cpu=24000  --wrap=\" python3 ComputeErrors.py " + str(problem) + " " + str(n) + " " + str(m) + " " + str(main_folder) + " " + str(plot) + " \""
                else:
                    string_to_exec = "sbatch --time=24:00:00 -n 1 --mem-per-cpu=8000 --wrap=\" python3 PlotRad.py " + str(n) + " " + str(m) + " " + str(main_folder) + " " + str(plot) + " \""
            print(string_to_exec)
            os.system(string_to_exec)
