import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
import json


class Dataset(object):
    def __init__(self, edges_file, genres_file):
        self.edges_df = pd.read_csv(edges_file)
        self.edges_df = self.edges_df[:5000]  # test # limit the dataset for quick CNM testing
        with open(genres_file) as json_file:
            self.genres = json.load(json_file)

    def edges(self):
        return self.edges_df

    def genres(self):
        return self.genres


def get_input():
    while True:
        print("Enter a node_id: ")
        try:
            node_id = int(input())
            if node_id >= 0:
                return node_id
            else:
                print("No negative numbers allowed")
        except ValueError as ve:
            print("Only enter numbers!")
        except Exception as e:
            print(e)
            return None


class CBFM(object):
    def __init__(self, genres):
        self.genres = genres
        self.sorted_friends_degree_gss = []
        self.selected_node_genre_set = ()
        if str(selected_node) in genres:
            self.selected_node_genre_set = set(genres[str(selected_node)])

    # Get each filtered friend genre set
    def get_top_selected_node_community_degree_GSS(self, top_selected_community_normalized_degrees):
        friends_degree_gss = {}
        for f in top_selected_community_normalized_degrees:
            friend_genre_set = ()
            if str(f[0]) in self.genres:
                friend_genre_set = set(self.genres[str(f[0])])
            # the intersection of the pair
            inter_size = len(self.selected_node_genre_set.intersection(friend_genre_set))
            # The Union of the pair
            union_size = len(self.selected_node_genre_set.union(friend_genre_set))
            # Calculate GSS
            GSS = inter_size / union_size
            # Add to dictionary
            friends_degree_gss[f] = GSS
        # sorted_friends_degree_gss = dict(sorted(friends_degree_gss.items(), key=lambda x: x[1], reverse=True))
        self.sorted_friends_degree_gss = sorted(friends_degree_gss.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTOP Selected Friends - Normalized Degree - GSS: {self.sorted_friends_degree_gss}")
        write_output(f"\n\nTOP Selected Friends - Normalized Degree - GSS: {self.sorted_friends_degree_gss}")
        return self.sorted_friends_degree_gss


# Network-Based Filtering Method
class NBFM(object):
    def __init__(self, graph):
        # STAGE 1: CNM Algorithm
        self.graph = graph


# Clauset-Newman-Moore Algorithm
class CNM(object):
    def __init__(self):
        self.communities = greedy_modularity_communities(self.graph)
        self.selected_community = []

    def get_selected_node_community(self, selected_node):
        for c in self.communities:
            if selected_node in c:
                self.selected_community.extend(c)
                break
        print(f'\nSelected Community: {self.selected_community}')
        write_output(f'\n\nSelected Community: {self.selected_community}')
        return self.selected_community


# Degree Centrality
class DC(object):
    def __init__(self, graph):
        self.degree_centrality = nx.degree_centrality(graph)
        # Filter the nodes only in the selected_community
        self.selected_community_degrees = {}
        self.top_selected_community_degrees = []
        self.top_selected_community_normalized_degrees = []

    def get_selected_node_community_degree(self, selected_community):
        # Filter the nodes only in the selected_community
        for n in selected_community:
            if n in self.degree_centrality:
                self.selected_community_degrees[n] = self.degree_centrality[n]
        print(f'\nSelected Community - Degrees: {self.selected_community_degrees}')
        write_output(f'\n\nSelected Community - Degrees: {self.selected_community_degrees}')
        # Remove the selected_node from the selected_community_degrees dictionary
        self.selected_community_degrees.pop(selected_node)
        # select the top 10 selected_community by degree
        # sorted_selected_community_degrees = sorted(selected_community_degrees.items(), key=lambda x: x[1])
        # print(f'Sorted Selected Community Degrees: {sorted_selected_community_degrees}')
        return self.selected_community_degrees

    def get_top_selected_node_community_degree(self, size=None, normalize=True):
        self.top_selected_community_degrees = sorted(self.selected_community_degrees.items(), key=lambda x: x[1], reverse=True)
        if size is not None:
            self.top_selected_community_degrees = self.top_selected_community_degrees[:size]
        print(f'\nTOP Selected Friends - Degrees: {self.top_selected_community_degrees}')
        write_output(f'\n\nTOP Selected Friends - Degrees: {self.top_selected_community_degrees}')
        # Normalizing the degree-centralities by dividing with the highest degree-centrality in the community
        if normalize:
            highest_degree_centrality = self.top_selected_community_degrees[0][1]
            self.top_selected_community_normalized_degrees = []
            for n in self.top_selected_community_degrees:
                n = list(n)
                n[1] /= highest_degree_centrality
                self.top_selected_community_normalized_degrees.append(tuple(n))
            print(f'\nTOP Selected Friends - Normalized Degrees: {self.top_selected_community_normalized_degrees}')
            write_output(f'\n\nTOP Selected Friends - Normalized Degrees: {self.top_selected_community_normalized_degrees}')
            return self.top_selected_community_normalized_degrees
        return self.top_selected_community_degrees


def calculate_IS_get_list(top_selected_node_community_degree_GSS, size=None):
    friends_influence_score = {}
    # Calculate Influence Score for each friend
    for n in top_selected_node_community_degree_GSS:
        # ((GSS * 2) + degree-centrality (DC)) / 3
        friends_influence_score[n[0][0]] = ((n[1] * 2) + n[0][1]) / 3
    sorted_friends_influence_score = sorted(friends_influence_score.items(), key=lambda x: x[1], reverse=True)
    if size is not None:
        sorted_friends_influence_score = sorted_friends_influence_score[:size]
    return dict(sorted_friends_influence_score)


def write_output(txt):
    with open("output.txt", "a") as f:
        f.write(txt)


if __name__ == '__main__':
    # Get the Dataset
    dataset = Dataset("edges.csv", "genres.json")

    # Input
    selected_node = get_input()

    # Import the graph
    G = nx.from_pandas_edgelist(dataset.edges(), "node_1", "node_2", create_using=nx.Graph)

    # STAGE 1: CNM Algorithm
    # selected_community
    cnm = CNM()
    selected_community = cnm.get_selected_node_community(selected_node)

    # STAGE 2: Degree-Centrality
    # top_selected_community_normalized_degrees
    dc = DC(G)
    selected_node_community_degree = dc.get_selected_node_community_degree(selected_community)
    top_selected_community_normalized_degrees = dc.get_top_selected_node_community_degree()

    # STAGE 3: CBFM
    # sorted_friends_degree_gss
    with open("genres.json") as json_file:
        genres = json.load(json_file)
    # genres = dataset.genres()
    cbfm = CBFM(genres)
    sorted_friends_degree_gss = cbfm.get_top_selected_node_community_degree_GSS(top_selected_community_normalized_degrees)
    '''
    node 1: ["Rap/Hip Hop", "Alternative", "Latin Music", "Pop"]
    node 43758: ["Dance", "Rap/Hip Hop", "Latin Music", "Pop", "Rock"]
    3 / 6 = 0.5
    '''
    # STAGE 4: Calculate Influence Score (IS)
    sorted_friends_influence_score = calculate_IS_get_list(sorted_friends_degree_gss, size=10)
    print(f"\nTOP Selected Friends - Influence Score: {sorted_friends_influence_score}")
    write_output(f"\n\nTOP Selected Friends - Influence Score: {sorted_friends_influence_score}")
