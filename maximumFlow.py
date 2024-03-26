from collections import defaultdict

class FordFulkerson:
    def __init__(self, graph, source, sink):
        self.graph = graph
        self.source = source
        self.sink = sink
        self.visited = set()

    def bfs(self, parent):
        queue = [self.source]
        self.visited.add(self.source)
        parent[self.source] = -1

        while queue:
            u = queue.pop(0)
            for v in range(len(self.graph)):
                if v not in self.visited and self.graph[u][v] > 0:
                    queue.append(v)
                    self.visited.add(v)
                    parent[v] = u
                    if v == self.sink:
                        return True
        return False

    def findMaxFlow(self):
        residualGraph = [row[:] for row in self.graph]
        maxFlow = 0
        parent = [-1] * len(self.graph)

        while self.bfs(parent):
            pathFlow = float('inf')
            v = self.sink
            while v != self.source:
                u = parent[v]
                pathFlow = min(pathFlow, residualGraph[u][v])
                v = u

            v = self.sink
            while v != self.source:
                u = parent[v]
                residualGraph[u][v] -= pathFlow
                residualGraph[v][u] += pathFlow
                v = u

            maxFlow += pathFlow

        return maxFlow


def solveDuplicateCattle(L, R, edges):
    source = 0
    sink = len(L) + len(R) + 1
    graph = [[0] * (len(L) + len(R) + 2) for _ in range(len(L) + len(R) + 2)]

    for lIndex, lNode in enumerate(L):
        graph[source][lIndex + 1] = 1

    for rIndex, rNode in enumerate(R):
        graph[rIndex + len(L) + 1][sink] = 1

    for edge in edges:
        lNode, rNode, weight = edge
        capacity = int(weight * 100)
        graph[L.index(lNode) + 1][R.index(rNode) + len(L) + 1] = capacity

    fordFulkerson = FordFulkerson(graph, source, sink)
    maxFlow = fordFulkerson.findMaxFlow()

    chosenEdges = []
    chosenRNodes = set()

    for lNode in L:
        maxWeight = 0
        maxRNode = -1
        for rNode in R:
            if graph[L.index(lNode) + 1][R.index(rNode) + len(L) + 1] > 0 and rNode not in chosenRNodes:
                weight = graph[L.index(lNode) + 1][R.index(rNode) + len(L) + 1] / 100.0
                if weight > maxWeight:
                    maxWeight = weight
                    maxRNode = rNode
        if maxRNode != -1:
            chosenEdges.append((lNode, maxRNode, maxWeight))
            chosenRNodes.add(maxRNode)

    return maxFlow, chosenEdges




def main():

    # Example usage
    L = [0, 1, 2]
    R = [3, 4, 5, 6]
    edges = [
        (0, 3, 0.5456),
        (0, 4, 0.455),
        (0, 5, 0.352),
        (0, 6, 0.1272),
        (1, 3, 0.840),
        (1, 4, 0.28),
        (1, 5, 0.751),
        (1, 6, 0.37),
        (2, 3, 0.4586),
        (2, 4, 0.590),
        (2, 5, 0.102),
        (2, 6, 0.5095),
    ]

    maxFlow, chosenEdges = solveDuplicateCattle(L, R, edges)
    print("Maximum flow:", maxFlow)
    print("Chosen edges:")
    for edge in chosenEdges:
        print("L node:", edge[0], ", R node:", edge[1], ", Weight:", edge[2])