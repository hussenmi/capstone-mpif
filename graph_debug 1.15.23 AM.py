# when creating the CellularGraphDataset data container, we had problems, so investigating it here
# it turns out that graph pickle files df_0_3_15.gpkl, df_2_1_6, and df_3_2_14.gpkl are the problmes when creating the data container.
# when we use the pickle files and create the graph (let's say G) and inspect G.nodes, for these two it goes like [1,2,3,...], but for others, it starts with a 0 ([0,1,2,3,...])
# therefore, they are the problems. let's try to create the graph pickle files again and see if that fixes the problem.
# even after creating the graph pickle files again, the problem persists.
# then i checked if the files associated with these graphs (cell-coords, cell_types, expression, voronoi json) have problems, but they just seem like the other ones.
# i then tried to delete the voronoi json files for these regions and create them again. even after doing that, there is the same problem. the graphs created from these files just don't have nodes[0]
# finally, i just decided to remove these three files instead, so they were not used when creating the `CellularGraphDataset` container.
# i also removed them from the graph labels dataset as well


import pickle
import networkx as nx
import os

# load the graph from the pickle file they provided (found in '../../space-gm/data/example_dataset/graph')
folder1 = "/juno/work/shah/users/ibrahih3/codebase/space-gm/data/example_dataset/graph"
pickle_graphs1 = os.listdir(folder1)
pickle_graphs1

# load the first pickle graph in the list
G1 = pickle.load(open(folder1 + "/" + pickle_graphs1[0], "rb"))

for i, g_f in enumerate(pickle_graphs1):
    G = pickle.load(open(folder1 + "/" + g_f, 'rb'))
    
    if 'cell_type' not in G.nodes[0]:
        print(i, g_f)
    else:
        # print(G.nodes[0])
        print('all_good')

# visualize the graph
# nx.draw(G1, with_labels=True)

# load the second pickle graph in the list
folder2 = "/juno/work/shah/users/ibrahih3/codebase/space-gm/example_data_melanoma/graph"
pickle_graphs2 = os.listdir(folder2)
pickle_graphs2

# load the first pickle graph in the list
G2 = pickle.load(open(folder2 + "/" + pickle_graphs2[0], "rb"))

for i, g_f in enumerate(pickle_graphs2):
    G = pickle.load(open(folder2 + "/" + g_f, 'rb'))
    if 0 not in G.nodes:
        print(i, g_f)
    else:
        # print(i, g_f)
        print('all_good')