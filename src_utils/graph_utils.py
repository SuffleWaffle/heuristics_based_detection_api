from itertools import chain, combinations
import networkx as nx
import igraph as ig
from src_utils.geometry_utils import is_part_of_other, check_line_type, euclidean_dist, find_vect_direct_rads, \
    get_n_closest, center_start_end_point_dist


def create_graph_all_cases(lines, top_n_to_cluster=100,
                           to_igraph=False,
                           add_point_eq_edges=True):
    # method 1
    G = create_graph(lines, add_edges=add_point_eq_edges)

    # divide into three different categories
    h_lines = [c for c, line in enumerate(lines) if check_line_type(line) == 'horizontal']
    v_lines = [c for c, line in enumerate(lines) if check_line_type(line) == 'vertical']
    o_lines = [c for c, line in enumerate(lines) if check_line_type(line) == 'other']

    # check for h_lines
    d_h_lines = {}
    for i in range(len(lines)):
        if i in h_lines:
            h_lines_container = d_h_lines.get(lines[i][1], set())
            h_lines_container.add(i)
            d_h_lines[lines[i][1]] = h_lines_container

    for i in range(len(h_lines)):
        idx1 = h_lines[i]
        line_i = lines[idx1]
        to_compare = d_h_lines[line_i[1]]
        for idx2 in to_compare:
            if idx1 != idx2:
                if is_part_of_other(lines[idx1], lines[idx2], ['horizontal']) or \
                        is_part_of_other(lines[idx2], lines[idx1], ['horizontal']):
                    G.add_edge(idx1, idx2)

    # check for v_lines
    d_v_lines = {}
    for i in range(len(lines)):
        if i in v_lines:
            v_lines_container = d_v_lines.get(lines[i][0], set())
            v_lines_container.add(i)
            d_v_lines[lines[i][0]] = v_lines_container

    for i in range(len(v_lines)):
        idx1 = v_lines[i]
        line_i = lines[idx1]
        to_compare = d_v_lines[line_i[0]]
        for idx2 in to_compare:
            if idx1 != idx2:
                if is_part_of_other(lines[idx1], lines[idx2], ['vertical']) or \
                        is_part_of_other(lines[idx2], lines[idx1], ['vertical']):
                    G.add_edge(idx1, idx2)

    # get radians
    rads = []
    for idx in o_lines:
        rads.append([find_vect_direct_rads(lines[idx]), \
                     find_vect_direct_rads([*lines[idx][2:], *lines[idx][:2]])])

    # check for o_lines
    o_lines_ = [lines[i] for i in o_lines]
    if o_lines_:
        _, candidates_indices = get_n_closest(o_lines_, None, center_start_end_point_dist,
                                              n=top_n_to_cluster,
                                              return_indices=True)
        for i in range(len(o_lines)):
            idx1 = o_lines[i]
            rads_i = rads[i]
            for j in candidates_indices[i]:
                if j != i:
                    idx2 = o_lines[j]
                    rads_j = rads[j]
                    if is_part_of_other(lines[idx1], lines[idx2], [], rads1=rads_i,
                                        rads2=rads_j) and \
                            is_part_of_other(lines[idx2], lines[idx1], [], rads1=rads_j,
                                             rads2=rads_i):
                        G.add_edge(idx1, idx2)
    if to_igraph:
        G = ig.Graph.from_networkx(G)
    return G

def create_graph(lines, to_igraph=False,
                 add_edges=True):
    # graph creation
    G = nx.Graph()
    for i, line in enumerate(lines):
        G.add_node(i)

    if add_edges:
        # create lines dict
        dict_w_lines = {}
        for i, line in enumerate(lines):
            start, end = tuple(line[:2]), tuple(line[2:])
            pos = dict_w_lines.get(start, [])
            pos.append(i)
            dict_w_lines[start] = pos

            pos = dict_w_lines.get(end, [])
            pos.append(i)
            dict_w_lines[end] = pos

        for to_connect in dict_w_lines.values():
            to_connect = list(combinations(to_connect, 2))
            G.add_edges_from(to_connect)

    G = G.to_undirected()
    if to_igraph:
        G = ig.Graph.from_networkx(G)
    return G


def create_graph_points(lines):# Check logic for continous not staight lines
    # Create an empty graph
    G = nx.Graph()
    # Add nodes to the graph
    for line in lines:
        G.add_node((line[0], line[1]), pos=(line[0], line[1]))
        G.add_node((line[2], line[3]), pos=(line[2], line[3]))
    # Add edges to the graph
    for line in lines:
        G.add_edge((line[0], line[1]), (line[2], line[3]))
    return G

def find_cyclic_objects(lines):# Find cyclic objects
    lines = [i for c, i in enumerate(lines)]
    lines = list(set(lines))

    G = create_graph_points(lines)

    G.remove_edges_from(list(nx.selfloop_edges(G)))

    cyclic = find_cyclic_subgraphs(G)

    possible_cycles = []
    for i in cyclic:
        if len(i.nodes()) > 1 and len(i.nodes()) < 500:
            possible_cycles.append(list(i.edges()))
    return possible_cycles

def find_cyclic_subgraphs(G):
    cyclic_subgraphs = []
    cycle_basis = nx.cycle_basis(G)

    for cycle in cycle_basis:
        subgraph = G.subgraph(cycle)
        cyclic_subgraphs.append(subgraph)

    return cyclic_subgraphs

def find_next_lines_for_grids(lines, start_line, center_hgrid):
    next_lines = []
    graph_lines_cand_for_turn = create_graph_points(lines)

    if euclidean_dist(start_line[:2], center_hgrid) < euclidean_dist(start_line[2:], center_hgrid):
        start_point = tuple(start_line[:2])
        end_point = tuple(start_line[2:])
    else:
        start_point = tuple(start_line[2:])
        end_point = tuple(start_line[:2])

    neighbors = list(graph_lines_cand_for_turn.neighbors(tuple(end_point)))
    if start_point in neighbors:
        neighbors.remove(start_point)
    if neighbors:
        next_lines.extend(neighbors)
        while neighbors:
            point_to_check = neighbors.pop()
            new_neighbors = list(graph_lines_cand_for_turn.neighbors(point_to_check))
            new_neighbors = [i for i in new_neighbors if i not in next_lines and \
                             i not in [start_point, end_point]]
            if not new_neighbors:
                break
            neighbors.extend(new_neighbors)
            next_lines.extend(new_neighbors)

    return next_lines


def merge_small_lines(lines, add_point_eq_edges=True):
    # define graph
    G = create_graph_all_cases(lines, add_point_eq_edges=add_point_eq_edges)
    # get big_lines
    big_lines = []
    to_del = []
    for component in nx.connected_components(G):
        big_line = []
        to_del_tmp = []
        for node in component:
            big_line.append(lines[node])
            to_del_tmp.append(node)

        if len(big_line) > 1:
            to_del.extend(to_del_tmp)
            big_lines.append(big_line)

    # get minimal and maximal point for each line
    actual_lines = []
    for line in big_lines:
        all_points = list(chain(*[[i[:2], i[2:]] for i in line]))
        max_point = max(all_points)
        min_point = min(all_points)
        actual_lines.append([*min_point, *max_point])

    return [i for c, i in enumerate(lines) if c not in to_del] + actual_lines


def merge_small_lines_all(lines):
    v_lines = [i for i in lines if check_line_type(i) == 'vertical']
    o_lines = [i for i in lines if check_line_type(i) == 'other']
    h_lines = [i for i in lines if check_line_type(i) == 'horizontal']
    return merge_small_lines(h_lines) \
        + merge_small_lines(v_lines) + merge_small_lines(o_lines, add_point_eq_edges=False)
