import os

for yaml in ["s_exp1_640f1","s_exp1_640f2", "s_exp1_640f3"]:
    os.system(f"CUDA_VISIBLE_DEVICES=3 python main.py --config expconfigs/{yaml}.yaml")