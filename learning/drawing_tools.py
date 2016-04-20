#import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def draw_sequences(Yh, Ah, Rh, Sh):
    f = plt.figure()
    spy = f.add_subplot(511)
    spy.stem(Yh[-100:], label="observations", linefmt='b-')
    plt.ylabel('Observations')
    plt.xlabel('step')
    
    spa = f.add_subplot(512)
    spa.stem(Ah[-100:], linefmt='r-', markerfmt='ro')
    plt.ylabel('Actions')
    plt.xlabel('step')
    
    spr = f.add_subplot(513)
    spr.stem(Rh[-100:], linefmt='g-', markerfmt='go')
    plt.ylabel('Rewards')
    plt.xlabel('step')
    
    sps = f.add_subplot(514)
    sps.stem(Sh[-100:], linefmt='k-', markerfmt='ko')
    plt.ylabel('State')
    plt.xlabel('step')
    
    f.show()

def draw_history(Q0h_s, Q0h_mc, Q0h_q, Q1h_s, Q1h_mc, Q1h_q, Alphah, Epsh):
    fq = plt.figure()
    line_Q0h, = plt.plot(Q0h_s, color='r', label="SARSA")
    plt.plot(Q0h_mc, color='b', label="Monte Carlo")
    plt.plot(Q0h_q, color='g', label="Q-Learning")
    plt.title('State-Action pair: [0,1] = 0')
    plt.ylabel('Q-Value')
    plt.xlabel('Step')
    plt.legend(handler_map={line_Q0h: HandlerLine2D(numpoints=4)})
    
    fqmc = plt.figure()
    line_Q1h, = plt.plot(Q1h_s, color='r', label="SARSA")
    plt.plot(Q1h_mc, color='b', label="Monte Carlo")
    plt.plot(Q1h_q, color='g', label="Q-Learning")
    plt.title('State-Action pair: [2,1] = 0.33')
    plt.ylabel('Q-Value')
    plt.xlabel('Step')
    plt.legend(handler_map={line_Q1h: HandlerLine2D(numpoints=4)})
    
    falpha = plt.figure()
    line_alpha_epsilon, = plt.plot(Alphah, color='b', label="Alpha")
    plt.plot(Epsh, color='r', label="Epsilon")
    plt.title('Alpha/Epsilon')
    plt.ylabel('Alpha/Epsilon')
    plt.xlabel('Step')
    plt.legend(handler_map={line_alpha_epsilon: HandlerLine2D(numpoints=4)})

    fq.show()
    fqmc.show()
    falpha.show()

def draw_epsilons(Eh, Eh1, Eh2, Eh3):
    f = plt.figure()
    
    line, = plt.plot(Eh, color='r', label="Eps(2, 1)")
    plt.plot(Eh1, color='b', label="Eps(4, 0)")
    plt.plot(Eh2, color='g', label="Eps(5, 0)")
    plt.plot(Eh3, color='k', label="Eps(6, 2)")
    plt.title('Epsilon')
    plt.ylabel('Epsilon')
    plt.xlabel('Step')
    plt.legend(handler_map={line: HandlerLine2D(numpoints=4)})
    
    f.show()






#def draw_graph(graph, labels = None, graph_layout = 'shell',
#                             node_size = 1600, node_color = 'blue', node_alpha = 0.3,
#                             node_text_size = 12,
#                             edge_color = 'blue', edge_alpha = 0.3, edge_tickness = 1,
#                             edge_text_pos = 0.3, text_font = 'sans-serif'):
#
#    # create networkx graph
#    G = nx.DiGraph()
#
#    # add edges
#    for edge in graph:
#        G.add_edge(edge[0], edge[1])
#
#    # these are different layouts for the network you may try
#    # shell seems to work best
#    if graph_layout == 'spring':
#        graph_pos = nx.spring_layout(G)
#    elif graph_layout == 'spectral':
#        graph_pos = nx.spectral_layout(G)
#    elif graph_layout == 'random':
#        graph_pos = nx.random_layout(G)
#    else:
#        graph_pos = nx.shell_layout(G)
#
#    # draw graph
#    nx.draw_networkx_nodes(G, graph_pos, node_size = node_size,
#                                                   alpha = node_alpha, node_color = node_color)
#
#    nx.draw_networkx_edges(G, graph_pos, width = edge_tickness,
#                                                  alpha = edge_alpha, edge_color = edge_color)
#
#    nx.draw_networkx_labels(G, graph_pos, font_size = node_text_size,
#                                                  font_family = text_font)
#
#    if labels is None:
#        labels = range(len(graph))
#
#    edge_labels = dict(zip(graph, labels))
#
#    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
#                                                              label_pos=edge_text_pos)
#
#    # show graph
#    plt.show()

#graph = [(0, 1), (1, 5), (1, 7), (4, 5), (4, 8), (1, 6), (3, 7), (5, 9),
#         (2, 4), (0, 4), (2, 5), (3, 6), (8, 9)]
#
## you may name your edge labels
#labels = map(chr, range(65, 65+len(graph)))
##draw_graph(graph, labels)
#
## if edge labels is not specified, numeric labels (0, 1, 2...) will be used
#draw_graph(graph)
