"""Implement random graph generators and Graph class."""

import random as ran
import copy
import numpy as np
import jacinle.random as random
import json
import openai
from sklearn.metrics import f1_score
import os
from openai import AzureOpenAI, OpenAI

__all__ = ['Graph', 'randomly_generate_graph_er', 'randomly_generate_graph_dnc',
           'get_random_graph_generator']


class Graph(object):
    """Store a graph using adjacency matrix.

    Args:
    nr_nodes: The number of nodes in the graph.
    edges: The adjacency matrix of the graph.
    """

    def __init__(self, nr_nodes, edges, coordinates=None):
        edges = edges.astype('int32')
        assert edges.min() >= 0 and edges.max() <= 1
        self._nr_nodes = nr_nodes
        self._edges = edges
        self._coordinates = coordinates
        self._shortest = None
        self.extra_info = {}

    @property
    def nr_nodes(self):
        return self._nr_nodes

    def get_edges(self):
        return copy.copy(self._edges)

    def get_coordinates(self):
        return self._coordinates

    def get_relations(self):
        """Return edges and identity matrix."""
        return np.stack([self.get_edges(), np.eye(self.nr_nodes)], axis=-1)

    def has_edge(self, x, y):
        return self._edges[x, y] == 1

    def get_out_degree(self):
        """Return the out degree of each node."""
        return np.sum(self._edges, axis=1)

    def get_shortest(self):
        """Return the length of shortest path between nodes."""
        if self._shortest is not None:
            return self._shortest

        n = self.nr_nodes
        edges = self.get_edges()

        # n + 1 indicates unreachable.
        shortest = np.ones((n, n)) * (n + 1)
        shortest[np.where(edges == 1)] = 1
        # Make sure that shortest[x, x] = 0
        shortest -= shortest * np.eye(n)
        shortest = shortest.astype('int32')

        # Floyd Algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        shortest[i, j] = min(shortest[i, j],shortest[i, k] + shortest[k, j])
        self._shortest = shortest
        return self._shortest

    def get_connectivity(self, k=None, exclude_self=True):
        """Calculate the k-connectivity.

        Args:
          k: The limited steps. unlimited if k=None or k<0.
          exclude_self: remove connectivity[x, x] if exclude_self=True.
        Returns:
          A numpy.ndarray representing the k-connectivity for each pair of nodes.
        """
        shortest = self.get_shortest()
        if k is None or k < 0:
            k = self.nr_nodes
        k = min(k, self.nr_nodes)
        conn = (shortest <= k).astype('int32')
        if exclude_self:
            n = self.nr_nodes
            inds = np.where(~np.eye(n, dtype=bool))
            conn = conn[inds]
            conn.resize(n, n - 1)
        return conn


def randomly_generate_graph_er(n, p, directed=False):
    """Randomly generate a graph by sampling the existence of each edge.

    Each edge between nodes has the probability $p (directed) or
    1 - (1-$p)^2 (undirected) to exist.

    Args:
    n: The number of nodes in the graph.
    p: the probability that a edge doesn't exist in directed graph.
    directed: Directed or Undirected graph. Default: False (undirected)

    Returns:
    A Graph class representing randomly generated graph.
    """
    edges = (random.rand(n, n) < p).astype('float')
    edges -= edges * np.eye(n)
    if not directed:
        edges = np.maximum(edges, edges.T)
    return Graph(n, edges)


def randomly_generate_graph_dnc(n, p=None, directed=False):
    """Random graph generation method as in DNC.

    As described in Differentiable neural computers (DNC),
    (https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)
    Sample $n nodes in a unit square. Then sample out-degree (m) of each nodes,
    connect to $m nearest neighbors (Euclidean distance) in the unit square.

    Args:
        n: The number of nodes in the graph.
        p: Control the sampling of the out-degree.
        If p=None, the default range is [1, n // 3].
        If p is float, the range is [1, int(n * p)].
        If p is int, the range is [1, p].
        If p is tuple. the range is [p[0], p[1]].
    directed: Directed or Undirected graph. Default: False (undirected)

    Returns:
    A Graph class representing randomly generated graph.
    """
    edges = np.zeros((n, n), dtype='float')
    pos = random.rand(n, 2)

    def dist(x, y):
        return ((x - y)**2).mean()

    if isinstance(p, tuple):
        lower, upper = p
    else:
        lower = 1
        if p is None:
            upper = n // 3
        elif isinstance(p, int):
            upper = p
        elif isinstance(p, float):
            upper = int(n * p)
        else:
            assert False, 'Unknown argument type: {}'.format(type(p))
        upper = max(upper, 1)
    lower = max(lower, 1)
    upper = min(upper, n - 1)

    for i in range(n):
        d = []
        k = random.randint(upper - lower + 1) + lower
        for j in range(n):
            if i != j:
                d.append((dist(pos[i], pos[j]), j))
        d.sort()
        for j in range(k):
            edges[i, d[j][1]] = 1
    if not directed:
        edges = np.maximum(edges, edges.T)
    return Graph(n, edges, pos)


def get_random_graph_generator(name):
    if name == 'dnc':
        return randomly_generate_graph_dnc
    return randomly_generate_graph_er

def generate_graph(n):
    graph = randomly_generate_graph_er(n, 0.15, directed=True)
    edge_graph = graph.get_edges()
    shortest_graph = graph.get_shortest()
    shortest_graph[shortest_graph>n]=-1
    connect_graph = graph.get_connectivity()
    return edge_graph, shortest_graph, connect_graph


def generate_relations(n, graph_array):
    statements_arr = []

    for i in range(n):
        for j in range(n):
            if graph_array[i,j] == 1:
                statements_arr.append("N"+str(i)+" has a directed edge to N"+str(j)+".")
            
    ran.shuffle(statements_arr)

    return ' '.join(statements_arr)


def create_system_prompt():
    return """
    The user will provide a question performing reasoning on a graph.
    Please give some reasoning steps briefly and give your matrix in a list of list in JSON format. 

    EXAMPLE JSON OUTPUT in square brackets:
    {
        "Brief Reasoning Steps": "Some brief reasoning steps.",
        "Matrix": [[1, 2, 3], [-1, 2, 3], [1, -1, 2]],"
    }
    """

def create_user_prompt_connectivity(n, relations):
    return "You are an agent who determines the relations in a graph. \
    For a directed graph containing " + str(n) + " nodes, \
    which is depicted as: " + relations + \
    " Now, from the above-given facts, you have to determine connectivity between the nodes. \
    You must give the reasoning process and you must give the final answer in a the format of a " + str(n) + "-by-" + str(n) + " matrix, \
    where the i,j-th entry is 1 if there is a path from Ni to Nj, 0 otherwise. \
    If there is no path from Ni to Nj, then the i,j-th entry is -1. \
    Question: What is connectivity between the nodes? \
    I need the matrix for further processing. Do not include anything other than the matrix in python-readable format in 'Matrix'. \
    Answer: "

def create_user_prompt_shortest(n, relations):
    return "You are an agent who determines the relations in a graph. \
    For a directed graph containing " + str(n) + " nodes, \
    which is depicted as: " + relations + \
    " Now, from the above-given facts, you have to determine the length of shortest paths between the nodes. \
    You must give the reasoning process and you must give the final answer in a the format of a " + str(n) + "-by-" + str(n) + " matrix, \
    where the i,j-th entry refers to the length of the shortest path from Ni to Nj. \
    If there is no path from Ni to Nj, then the i,j-th entry is -1. \
    Question: What is the length of shortest paths between the nodes? \
    I need the matrix for further processing. Do not include anything other than the matrix in python-readable format in 'Matrix'. \
    Answer: "


def create_client_azureOpenAI(api_version, api_key, azure_endpoint):
    return AzureOpenAI(
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=azure_endpoint
    )


def create_client_ds(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)


def prompt_llm_ds(client, sys_prompt, user_prompt, llm_model = "deepseek-chat"):
    response = client.chat.completions.create(
        model=llm_model,  
        max_tokens=8192,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        response_format={
            'type': 'json_object'
        }
    )
    return response

def prompt_llm_ds_r1(client, sys_prompt, user_prompt, llm_model = "deepseek-reasoner"):
    response = client.chat.completions.create(
        model=llm_model,  
        max_tokens=8192,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    return response

    
def extract_json_and_numpy_array(llm_output):
    response = llm_output.choices[0].message.content
    response = response.replace("```json", "")
    response = response.replace("```", "")
    return json.loads(response), np.array(json.loads(response)['Matrix'])


def cal_f1_score(correct_matrix, output_matrix, weight_method='micro'):
    return f1_score(correct_matrix.ravel(), output_matrix.ravel(), average=weight_method)

def save_json_numpy(file_name, json_output, output_matrix):
    with open(file_name+'.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
