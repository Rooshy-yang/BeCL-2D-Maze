import os
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
os.environ["ROOT_DIR"] = ".."  # this allows to use relative paths in config files
from result_inspection.toy_maze import *
from result_inspection.plot_helpers import *

plot_kwargs = dict(stat_list=['cumulative_rew'], labels=['Reward'], figsize=(6, 4), titlesize=14)
skill_kwargs = dict(figsize=(5,5), reset_dict=dict(state=torch.tensor([0., -0.5])))


algo = "contrastive_mi"
env = "tree_maze_1"

exp, cmap = load_exp_data("{}/{}".format(env, algo), notebook_mode=False)
ax = plot_all_skills(exp, cmap, notebook_mode=False,  **skill_kwargs)
plt.savefig("../MI-result/images/"+algo+"_"+env+'.png', dpi=300)
