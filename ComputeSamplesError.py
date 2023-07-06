import os

problems = ["sine", "helm"]

noise = [0]

dictionary_m = {"sine": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "helm": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

for problem in problems:
    for n in noise:
        m_vec = dictionary_m[problem]
        print(problem, m_vec)
        for m in m_vec:
            string_to_exec = "sbatch --time=4:00:00 -n 1  --mem-per-cpu=24000  --wrap=\" python3 ComputeMoreSamples.py " + str(problem) + " " + str(n) + " " + str(m) + " \""
            os.system(string_to_exec)
