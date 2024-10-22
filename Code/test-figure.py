import pandas as pd
import matplotlib.pyplot as plt
import re

# Plot accuracy figure using provided data on Zenodo
def plot_probability_best_arm_fixed_sd(file_paths, best_arm):
    """
    Plot the probability of finding the best arm over time with different real standard deviations.
    
    Parameters:
    - file_paths: List of file paths to the CSV files containing simulation data.
    - best_arm: The index of the best arm (as known from the scenario).
    """
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Calculate the probability of selecting the best arm at each time step
        prob_best_arm = df.groupby('horizon')['chosen_arm'].apply(lambda x: (x == best_arm).mean())
        
        # Extract real SD from file path
        real_sd = file_path.split('_')[-1].replace('.csv', '')
        print(real_sd)
        
        # Plot
        plt.plot(prob_best_arm.index, prob_best_arm.values, label=f'Real SD={real_sd}')

    # Customize plot
    plt.xlabel('Time Horizon')
    plt.ylabel('Probability of Finding Best Arm')
    plt.title('Probability of Finding Best Arm over Time Using TS Gaussian Squared (fixed var = 1)')
    plt.legend(title='Real SD')
    plt.grid(True)

    # Show plot
    plt.show()

# Example usage
file_paths = [
    r'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario1\TS\TS_squared\realsd_0.1.csv',
    r'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario1\TS\TS_squared\realsd_0.5.csv',
    r'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario1\TS\TS_squared\realsd_0.25.csv',
    r'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario1\TS\TS_squared\realsd_1.csv',
    r'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario1\TS\TS_squared\realsd_1.5.csv'
]

best_arm = 4  # Replace this with the correct best arm index
#plot_probability_best_arm_fixed_sd(file_paths, best_arm)

## different assumed sd for TS/BayesUCB Gaussian squared, using self-tested data
def plot_probability_best_arm(realsd, best_arm, algo, scenario):
    """
    Plot the probability of finding the best arm over time with different assumed standard deviations.

    Parameters:
    - file_paths: List of file paths to the CSV files containing the data.
    - best_arm: The index of the best arm (as known from the scenario).
    """

    if algo == "TS":
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\TS\TS_squared'
        file_paths = [
            f'{base_dir}/ts_gaussian_squared_realsd_{realsd}_assumed_sd_{sd}.csv' for sd in [0.1, 0.25, 0.5, 0.75, 1, 1.5]
        ]
    elif algo == "Bayes UCB":
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_squared\NormalArm_c=2'
        file_paths = [
            f'{base_dir}/bayes_ucb_gaussian_squared_realsd_{realsd}_assumed_sd_{sd}.csv' for sd in [0.1, 0.25, 0.5, 0.75, 1, 1.5]
        ]
    else: # Bayes UCB PPF
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_PPF_squared\NormalArm_c=2'
        file_paths = [
            f'{base_dir}/bayes_ucb_gaussian_squared_realsd_{realsd}_assumed_sd_{sd}.csv' for sd in [0.1, 0.25, 0.5, 0.75, 1, 1.5]
        ]

    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Calculate the probability of selecting the best arm at each time step
        prob_best_arm = df.groupby('horizon')['chosen_arm'].apply(lambda x: (x == best_arm).mean())
        
        # Extract assumed SD from file path
        assumed_sd = file_path.split('_')[-1].replace('.csv', '')
        
        # Plot
        plt.plot(prob_best_arm.index, prob_best_arm.values, label=f'Assumed SD={assumed_sd}')

    # Customize plot
    plt.xlabel('Time Horizon')
    plt.ylabel('Probability of Finding Best Arm')
    plt.title('Actual SD={}'.format(realsd))
    plt.legend(title='Assumed SD')
    plt.grid(True)

    # Show plot
    plt.show()

best_arm = 4  # Replace with the correct best arm index

# Call the function
real_sd_values = [0.1, 0.25, 0.5, 0.75, 1, 1.5]

# Loop over each real_sd value and call the function
#for real_sd in real_sd_values:
 #   plot_probability_best_arm(real_sd, best_arm, "Bayes UCB PPF", 2)

#for real_sd in real_sd_values:
    #plot_probability_best_arm(real_sd, best_arm, "Bayes UCB", 2)

#for real_sd in real_sd_values:
    #plot_probability_best_arm(real_sd, best_arm, "TS", 2)


def plot_probability_best_arm_fixed_sd(assumed_sd, best_arm, algo, scenario):
    """
    Plot the probability of finding the best arm over time with different real standard deviations.

    Parameters:
    - realsd: The actual standard deviation in the simulations.
    - best_arm: The index of the best arm (as known from the scenario).
    - algo: The algorithm used ("TS", "Bayes UCB", or "Bayes UCB PPF").
    - assumed_sd: The assumed standard deviation used by the algorithm.
    """
    if algo == "TS":
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\TS\TS_squared'
        file_paths = [
            f'{base_dir}/ts_gaussian_squared_realsd_{realsd}_assumed_sd_{assumed_sd}.csv' for realsd in [0.1, 0.25, 0.5, 0.75, 1, 1.5]
        ]
    elif algo == "Bayes UCB":
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_squared\NormalArm_c=2'
        file_paths = [
            f'{base_dir}/bayes_ucb_gaussian_squared_realsd_{realsd}_assumed_sd_{assumed_sd}.csv' for realsd in [0.1, 0.25, 0.5, 0.75, 1, 1.5]
        ]
    else: # Bayes UCB PPF
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_PPF_squared\NormalArm_c=2'
        file_paths = [
            f'{base_dir}/bayes_ucb_gaussian_squared_realsd_{realsd}_assumed_sd_{assumed_sd}.csv' for realsd in [0.1, 0.25, 0.5, 0.75, 1, 1.5]
        ]

    plt.figure(figsize=(10, 6))

    real_sd = [0.1, 0.25, 0.5, 0.75, 1, 1.5]
    index=0
    for file_path in file_paths:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Calculate the probability of selecting the best arm at each time step
        prob_best_arm = df.groupby('horizon')['chosen_arm'].apply(lambda x: (x == best_arm).mean())
        real_sd_current = real_sd[index]
        index = index + 1
        
        # Plot
        plt.plot(prob_best_arm.index, prob_best_arm.values, label=f'Real SD={real_sd_current}')

    # Customize plot
    plt.xlabel('Time Horizon')
    plt.ylabel('Probability of Finding Best Arm')
    plt.title('Probability of Finding Best Arm over Time Using {} Gaussian Squared (fixed sd = {})'.format(algo, assumed_sd))
    plt.legend(title='Real SD')
    plt.grid(True)

    # Show plot
    plt.show()

best_arm = 4  # Replace this with the correct best arm index
#plot_probability_best_arm_fixed_sd(0.75, best_arm, "TS", 2)
#plot_probability_best_arm_fixed_sd(1, best_arm, "TS", 2)
#plot_probability_best_arm_fixed_sd(1.5, best_arm, "TS", 2)

#plot_probability_best_arm_fixed_sd(1, best_arm, "Bayes UCB", 2)
#plot_probability_best_arm_fixed_sd(1.5, best_arm, "Bayes UCB", 2)
#plot_probability_best_arm_fixed_sd(2, best_arm, "Bayes UCB", 1)

#plot_probability_best_arm_fixed_sd(1, best_arm, "Bayes UCB PPF", 2)
#plot_probability_best_arm_fixed_sd(1.5, best_arm, "Bayes UCB PPF", 2)


def plot_probability_best_arm_different_c(real_sd, assumed_sd, best_arm, algo, scenario):
    """
    Plot the probability of finding the best arm over time with different tunable parameter c in separate subplots.
    
    Parameters:
    - real_sd: The actual standard deviation in the simulations.
    - assumed_sd: The assumed standard deviation used by the algorithm.
    - best_arm: The index of the best arm (as known from the scenario).
    - algo: The algorithm used ("Bayes UCB" or "Bayes UCB PPF").
    """
    if algo == "Bayes UCB":
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_squared'
    else:
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_PPF_squared'
    
    c_values = [1, 1.5, 2, 2.5, 3]
    file_paths = [
        f'{base_dir}/NormalArm_c={c}/bayes_ucb_gaussian_squared_realsd_{real_sd}_assumed_sd_{assumed_sd}.csv' for c in c_values
    ]
    plt.figure(figsize=(10, 6))

    index=0

    for file_path in file_paths:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Calculate the probability of selecting the best arm at each time step
        prob_best_arm = df.groupby('horizon')['chosen_arm'].apply(lambda x: (x == best_arm).mean())
        c_current = c_values[index]
        index = index + 1
        
        # Plot
        plt.plot(prob_best_arm.index, prob_best_arm.values, label=f'c={c_current}')

    # Customize plot
    plt.xlabel('Time Horizon')
    plt.ylabel('Probability of Finding Best Arm')
    plt.title('Probability of Finding Best Arm over Time Using {} Gaussian Squared (fixed sd = {}, real sd = {})'.format(algo, assumed_sd, real_sd))
    plt.legend(title='c value')
    plt.grid(True)

    # Show plot
    plt.show()


####### same as the function above, just combine the figures into a single plot
def plot_probability_best_arm_different_c(real_sd, assumed_sd, best_arm, algo, ax, scenario):
    """
    Plot the probability of finding the best arm over time with different tunable parameter c in a given subplot.
    
    Parameters:
    - real_sd: The actual standard deviation in the simulations.
    - assumed_sd: The assumed standard deviation used by the algorithm.
    - best_arm: The index of the best arm (as known from the scenario).
    - algo: The algorithm used ("Bayes UCB" or "Bayes UCB PPF").
    - ax: The axis to plot on.
    """
    if algo == "Bayes UCB":
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_squared'
    else:
        base_dir = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_PPF_squared'
    
    c_values = [1, 1.5, 2, 2.5, 3]
    file_paths = [
        f'{base_dir}/NormalArm_c={c}/bayes_ucb_gaussian_squared_realsd_{real_sd}_assumed_sd_{assumed_sd}.csv' for c in c_values
    ]

    for index, file_path in enumerate(file_paths):
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Calculate the probability of selecting the best arm at each time step
        prob_best_arm = df.groupby('horizon')['chosen_arm'].apply(lambda x: (x == best_arm).mean())
        c_current = c_values[index]
        
        # Plot on the given axis
        ax.plot(prob_best_arm.index, prob_best_arm.values, label=f'c={c_current}')

    # Customize subplot
    ax.set_xlabel('Time Horizon')
    ax.set_ylabel('Probability of Finding Best Arm')
    ax.set_title(f'{algo} Gaussian Squared (fixed sd = {assumed_sd}, real sd = {real_sd})')
    ax.legend(title='c value')
    ax.grid(True)

"""
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8))

# Plotting the first 4 figures
plot_probability_best_arm_different_c(0.25, 1, best_arm, "Bayes UCB", axs1[0, 0], 2)
plot_probability_best_arm_different_c(0.25, 1.5, best_arm, "Bayes UCB", axs1[0, 1], 2)
plot_probability_best_arm_different_c(0.5, 1, best_arm, "Bayes UCB", axs1[1, 0], 2)
plot_probability_best_arm_different_c(0.5, 1.5, best_arm, "Bayes UCB", axs1[1, 1], 2)

plt.suptitle('Probability of Finding Best Arm over Time for Different Real and Assumed SDs', fontsize=16)
plt.tight_layout()
plt.show()

# Create the figure and axes for the second set of 4 plots
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

# Plotting the next 4 figures
plot_probability_best_arm_different_c(0.75, 1, best_arm, "Bayes UCB", axs2[0, 0], 2)
plot_probability_best_arm_different_c(0.75, 1.5, best_arm, "Bayes UCB", axs2[0, 1], 2)
plot_probability_best_arm_different_c(1, 1, best_arm, "Bayes UCB", axs2[1, 0], 2)
plot_probability_best_arm_different_c(1, 1.5, best_arm, "Bayes UCB", axs2[1, 1], 2)

plt.tight_layout()
plt.show()

########### Bayes UCB PPF
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8))

# Plotting the first 4 figures
plot_probability_best_arm_different_c(0.25, 1, best_arm, "Bayes UCB PPF", axs1[0, 0], 2)
plot_probability_best_arm_different_c(0.25, 1.5, best_arm, "Bayes UCB PPF", axs1[0, 1], 2)
plot_probability_best_arm_different_c(0.5, 1, best_arm, "Bayes UCB PPF", axs1[1, 0], 2)
plot_probability_best_arm_different_c(0.5, 1.5, best_arm, "Bayes UCB PPF", axs1[1, 1], 2)

plt.suptitle('Probability of Finding Best Arm over Time for Different Real and Assumed SDs', fontsize=16)
plt.tight_layout()
plt.show()

# Create the figure and axes for the second set of 4 plots
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

# Plotting the next 4 figures
plot_probability_best_arm_different_c(0.75, 1, best_arm, "Bayes UCB PPF", axs2[0, 0], 2)
plot_probability_best_arm_different_c(0.75, 1.5, best_arm, "Bayes UCB PPF", axs2[0, 1], 2)
plot_probability_best_arm_different_c(1, 1, best_arm, "Bayes UCB PPF", axs2[1, 0], 2)
plot_probability_best_arm_different_c(1, 1.5, best_arm, "Bayes UCB PPF", axs2[1, 1], 2)

plt.tight_layout()
plt.show()
"""

#######
# for same real sd, test accuracy of different algorithms
def algo_compare(realsd, best_arm, scenario):
    base_dir1 = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\TS\TS_squared'
    base_dir2 = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\TS\TS'
    base_dir3 = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_squared\NormalArm_c=2'
    base_dir4 = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian_PPF_squared\NormalArm_c=2'
    base_dir5 = rf'C:\Users\lpy20\Desktop\UCLA\ML in Chem\optimization data logs\optimization data logs\synthetic data simulation logs\normal arm\scenario{scenario}\BayesUCBGaussian\BayesUCBGaussian'

    file_paths = [
        f'{base_dir1}/ts_gaussian_squared_realsd_{realsd}_assumed_sd_1.csv',
        f'{base_dir2}/ts_gaussian_realsd_{realsd}_assumed_sd_0.25.csv',
        f'{base_dir2}/ts_gaussian_realsd_{realsd}_assumed_sd_0.5.csv',
        f'{base_dir3}/bayes_ucb_gaussian_squared_realsd_{realsd}_assumed_sd_1.csv',
        f'{base_dir4}/bayes_ucb_gaussian_squared_realsd_{realsd}_assumed_sd_1.csv',
        f'{base_dir5}/bayes_ucb_gaussian_realsd_{realsd}_assumed_sd_0.25.csv',
    ]

    plt.figure(figsize=(10, 6))

    algo = [
        "TS Gaussian Squared (fixed sd = 1)",
        "TS Gaussian (fixed sd = 0.25)",
        "TS Gaussian (fixed sd = 0.5)",
        "Bayes UCB Gaussian Squared (fixed sd = 1)",
        "Bayes UCB Gaussian Squared PPF (fixed sd = 1)",
        "Bayes UCB Gaussian (fixed sd = 0.25)"
    ]
    index=0
    for file_path in file_paths:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Calculate the probability of selecting the best arm at each time step
        prob_best_arm = df.groupby('horizon')['chosen_arm'].apply(lambda x: (x == best_arm).mean())
        algo_current = algo[index]
        index = index + 1
        
        # Plot
        plt.plot(prob_best_arm.index, prob_best_arm.values, label=f'{algo_current}')

    # Customize plot
    plt.xlabel('Time Horizon')
    plt.ylabel('Probability of Finding Best Arm')
    plt.title('Accuracy of Normal Reward Testing Best Performers, Scenario{}, Actual SD = {}'.format(scenario, realsd))
    plt.legend(title='Algorithm')
    plt.grid(True)

    # Show plot
    plt.show()

#algo_compare(0.1, best_arm = 4, scenario = 1)
#algo_compare(1.5, best_arm = 4, scenario = 1)
#algo_compare(0.1, best_arm = 4, scenario = 2)
#algo_compare(1.5, best_arm = 4, scenario = 2)

#algo_compare(0.25, best_arm = 4, scenario = 1)
#algo_compare(0.5, best_arm = 4, scenario = 1)
#algo_compare(0.75, best_arm = 4, scenario = 1)
#algo_compare(1, best_arm = 4, scenario = 1)


#algo_compare(0.25, best_arm = 4, scenario = 2)
#algo_compare(0.5, best_arm = 4, scenario = 2)
#algo_compare(0.75, best_arm = 4, scenario = 2)
#algo_compare(1, best_arm = 4, scenario = 2)
