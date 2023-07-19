import pandas as pd
import matplotlib.pyplot as plt

JOB_ID = 9253916

# parse .out file
with open(f'results/job-{JOB_ID}.out', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'max_fitnesses:' in line.lower():
            max_approx_fitnesses = eval(line.split(':')[-1].strip())
        elif 'mean_fitnesses:' in line.lower():
            mean_approx_fitnesses = eval(line.split(':')[-1].strip())
        elif 'median_fitnesses:' in line.lower():
            median_approx_fitnesses = eval(line.split(':')[-1].strip())

n_gens = 200

mean_fitnesses = [-0.09133421, -0.08143251, -0.0744701, -0.070269495, -0.06642679, -0.0623166, -0.060565695, -0.059021913, -0.057861198, -0.057125103, -0.055375258400140875, -0.06957910643242765, -0.07274841242002068, -0.07056895279915018, -0.07192733314018648]
median_fitnesses = [-0.089825004, -0.078265, -0.073805004, -0.06928, -0.06575, -0.06146, -0.059260003, -0.057405002, -0.057044998, -0.05641, -0.05359648649004625, -0.052893917355788606, -0.05215085761153718, -0.05207066106927445, -0.05219061805396208]
max_fitnesses = [-0.06395, -0.05502, -0.05561, -0.05462, -0.054, -0.04883, -0.04699, -0.05013, -0.04769, -0.04631, -0.04534501800663504, -0.04613143477987444, -0.04340251139929077, -0.04486692195817174, -0.044855841955299375]
min_fitnesses = [-0.13893, -0.11843, -0.09163, -0.10114, -0.09369, -0.08548, -0.08849, -0.08307, -0.08177, -0.08363, -0.08187331913374285, -0.19581133695010594, -0.1797424275654598, -0.18580451602830644, -0.2221732977180489]


# parse csv file
df = pd.read_csv(f'datasets/{JOB_ID}.csv')
for gen in range(n_gens + 1):
    gen_df = df[df['gen'] == gen]
    max_fitnesses = gen_df['fitness'].max()
    mean_fitnesses = gen_df['fitness'].mean()
    median_fitnesses = gen_df['fitness'].median()

plt.plot(mean_approx_fitnesses, label='approx mean')
# plt.plot(median_approx_fitnesses, label='approx median')
plt.plot(max_approx_fitnesses, label='approx max')
# plt.plot(min_approx_fitnesses, label='approx min')
plt.plot(mean_fitnesses, label='mean')
# plt.plot(median_fitnesses, label='median')
plt.plot(max_fitnesses, label='max')
# plt.plot(min_fitnesses, label='min')
plt.xlabel('generation')
plt.ylabel('fitness')
plt.xticks(range(0, len(mean_fitnesses) + 1, 5))

# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
plt.tight_layout()
plt.show()
