import operator
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from networkx.algorithms.community import greedy_modularity_communities
import json

'''
def set_node_community(G, communities):
    #Add community to node attributes
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


def set_edge_community(G):
    # Find internal edges and add their community to their attributes
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0


def get_color(i, r_off=1, g_off=1, b_off=1):
    # Assign a color to a vertex.
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)
'''

def get_dataset():
    # Use a breakpoint in the code line below to debug your script.
    edges = pd.read_csv("./edges.csv")
    print(edges)


if __name__ == '__main__':
    # get_dataset()
    edges_df = pd.read_csv("./edges.csv")
    df2 = edges_df[:20000]
    # print(df2)
    G = nx.from_pandas_edgelist(df2, "node_1", "node_2", create_using=nx.Graph)

    # Input
    while True:
        print("Enter a node_id: ")
        try:
            selected_node = int(input())
            if selected_node >= 0:
                break
            else:
                print("No negative numbers allowed")
        except ValueError as ve:
            print("Only enter numbers!")
        except Exception as e:
            print(e)
            exit(-1)

    ## STAGE 1: CNM Algorithm
    communities = greedy_modularity_communities(G)  # 200,000 took about an hour
    print(f'\nCommunities: {communities}')
    # Get the community of the selected node
    selected_community = []
    for c in communities:
        if selected_node in c:
            selected_community.extend(c)
            break
    print(f'\nSelected Community: {selected_community}')

    ## STAGE 2: Degree-Centrality
    centrality = nx.degree_centrality(G)
    # Filter the nodes only in the selected_community
    selected_community_degrees = {}
    for n in selected_community:
        if n in centrality:
            selected_community_degrees[n] = centrality[n]
    print(f'\nSelected Community Degrees: {selected_community_degrees}')
    # Remove the selected_node from the selected_community_degrees dictionary
    selected_community_degrees.pop(selected_node)
    # select the top 10 selected_community by degree
    # sorted_selected_community_degrees = sorted(selected_community_degrees.items(), key=lambda x: x[1])
    # print(f'Sorted Selected Community Degrees: {sorted_selected_community_degrees}')

    top_selected_community_degrees = sorted(selected_community_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    # top_selected_community_degrees = sorted(selected_community_degrees.items(), key=operator.itemgetter(1))
    print(f'\nTOP Selected Community Degrees: {top_selected_community_degrees}')

    ## STAGE 3: CBFM
    # Dataset
    with open('genres.json') as json_file:
        genres = json.load(json_file)
    # Get the selected_node genre list
    friends_degree_gss = {}
    snode_genre_set = set(genres[str(selected_node)])
    # Get each filtered friend genre set
    for f in top_selected_community_degrees:
        friend_genre_set = set(genres[str(f[0])])
        # the intersection of the pair
        inter_size = len(snode_genre_set.intersection(friend_genre_set))
        # The Union of the pair
        union_size = len(snode_genre_set.union(friend_genre_set))
        # Calculate GSS
        GSS = inter_size / union_size
        # Add to dictionary
        friends_degree_gss[f] = GSS

    sorted_friends_degree_gss = dict(sorted(friends_degree_gss.items(), key=lambda x: x[1], reverse=True))
    print(f"\nsorted_friends_degree_gss: {sorted_friends_degree_gss}")
    '''
    node 1: ["Rap/Hip Hop", "Alternative", "Latin Music", "Pop"]
    node 43758: ["Dance", "Rap/Hip Hop", "Latin Music", "Pop", "Rock"]
    3 / 6 = 0.5
    '''
    # Sort the degrees of the friends list
    # TODO: Bring the friends that have similar GSS values but higher degree-centrality, to the top of the list
    # TODO: Organize the functions and classes

    '''
    # Drawing
    pos = nx.spring_layout(G, k=0.1)
    plt.rcParams.update({'figure.figsize': (15, 10)})
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=0,
        edge_color="#444444",
        alpha=0.05,
        with_labels=False)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize': (15, 10)})
    plt.style.use('dark_background')
    # Set node and edge communities
    set_node_community(G, communities)
    set_edge_community(G)
    # Set community color for internal edges
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ["black" for e in internal]
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
    # external edges
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=0,
        edgelist=external,
        edge_color="silver",
        node_color=node_color,
        alpha=0.2,
        with_labels=False)
    # internal edges
    nx.draw_networkx(
        G, pos=pos,
        edgelist=internal,
        edge_color=internal_color,
        node_color=node_color,
        alpha=0.05,
        with_labels=False)
    '''




    # BFS - Finding nodes from the source
    '''
    bfs_list = list(nx.bfs_edges(G, source=selected_node, depth_limit=3))
    print(f'BFS List for source 0 = {bfs_list}')
    '''

    # Average Degree
    '''
    mean = np.mean([d for _, d in G.degree()])
    print(f'Average Degree = {mean}')
    '''

    # Eigenvector Centrality
    '''
    centrality = nx.eigenvector_centrality(G)
    c_list = sorted((v, f"{c:0.2f}") for v, c in centrality.items())
    # Highest Eigenvector value
    val = 0.0
    node = 0
    for c in c_list:
        if float(c[1]) > val:
            val = float(c[1])
            node = c[0]
    print(f'Node with highest eigenvector centrality = {node} : {val}')
    '''

    # Draw the graph
    '''
    nx.draw_networkx(G)  # 20,000 rows took about 15 min
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    '''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
