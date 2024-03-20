import os
from ete3 import Tree
import random
import pyvolve
from tqdm import tqdm
#默认物种数等于10、loci数目、DNA长度
species=10
loci=20
length=500

def generate_gtr_parameters():
            """
            Generates random but reasonable parameters for a GTR model, including state frequencies
            and a rate matrix. The state frequencies sum to 1, and the rate matrix is symmetric.

            Returns:
                dict: A dictionary containing 'state_freqs' and 'rate_matrix'.
            """

            # Generate random state frequencies that sum to 1
            state_freqs = [random.random() for _ in range(4)]
            total = sum(state_freqs)
            state_freqs = [freq / total for freq in state_freqs]

            # Generate a symmetric rate matrix with None on the diagonal
            rate_matrix = []
            for i in range(4):
                row = []
                for j in range(4):
                    if i == j:
                        row.append(None)
                    elif j < i:
                        # Copy the symmetric element
                        row.append(rate_matrix[j][i])
                    else:
                        # Generate a random rate
                        row.append(random.uniform(0.5, 3))  # Rates between 0.5 and 3
                rate_matrix.append(row)

            return {"state_freqs": state_freqs, "rate_matrix": rate_matrix}
#################################################################################################
def generate_random_trees(num, path):
    # 确保提供的路径存在
    if not os.path.exists(path):
        os.makedirs(path)

    # 使用 tqdm 添加进度条
    for i in tqdm(range(num), desc="Generating Trees"):
        # 直接创建一个无根树，并随机分配分支长度
        tree = Tree()
        tree.populate(species, names_library=["Sp{}".format(j) for j in range(1, 10 + 1)], random_branches=True)
    
        tree.unroot(mode='legacy')
        newick_tree = tree.write(format=1, format_root_node=True)

        # 保存树的Newick格式
        tree_file = os.path.join(path, f"tree_{i+1}.txt")
        with open(tree_file, 'w') as tf:
            tf.write(newick_tree)

def simulate_dna_sequences(tree_path, tree_num, output_path, num_loci, DNA_length,model_type='JC'):
    """
    Simulate DNA sequences for multiple loci on phylogenetic trees and save them to files with specific names.

    :param tree_path: The folder containing tree files in Newick format.
    :param tree_num: The number of tree files to read.
    :param output_path: The folder where output DNA files will be saved.
    :param num_loci: The number of loci to simulate for each tree.
    :param DNA_length: The length of each DNA locus.
    """

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create an outer tqdm loop for tree_idx
    for tree_idx in tqdm(range(1, tree_num + 1), desc="Simulating Trees"):
        tree_file_name = os.path.join(tree_path, f'tree_{tree_idx}.txt')
        if not os.path.exists(tree_file_name):
            print(f"Tree file '{tree_file_name}' not found. Skipping.")
            continue

        # Read the tree from the file
        with open(tree_file_name, 'r') as tree_file:
            newick_tree = tree_file.read()

        # Create a Pyvolve Tree object
        tree = pyvolve.read_tree(tree=newick_tree)
        
        if model_type=="JC":
            # Define the evolutionary model
            model = pyvolve.Model("nucleotide", {"kappa": 2.0})

            # Define partitions for DNA simulation
            partitions = [pyvolve.Partition(models=model, size=DNA_length) for _ in range(num_loci)]

        else:
            # Define the evolutionary model
            parameter = generate_gtr_parameters()
            model = pyvolve.Model("nucleotide", parameter)

            # Define partitions for DNA simulation
            partitions = [pyvolve.Partition(models=model, size=DNA_length) for _ in range(num_loci)]


        # Create an Evolver object to simulate sequence evolution for each loci
        for loci in range(num_loci):
            evolver = pyvolve.Evolver(tree=tree, partitions=partitions[loci])

            # Perform the simulation
            evolver(seqfile=os.path.join(output_path, f'tree{tree_idx}_{loci}_simulated_alignment.fasta'))
