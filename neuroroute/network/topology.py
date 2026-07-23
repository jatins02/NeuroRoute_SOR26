import json
import os
import yaml

class TopologyManager:
    def __init__(self):
        # graph format: {source: {dest: {"latency" : float, "bandwidth" : float}}}
        self.graph : dict[str, dict[str, dict[float, float]]] = {}

    def load_topology(self, location) -> None:
        # check if filepath exists
        if not os.path.exists(location):
            raise FileNotFoundError(f"Filepath {location} doesn't exist.")

        # get extension of filepath
        ext = os.path.splitext(location)[1].lower()
        # print(ext)

        # open file and load data according to extension
        with open(location, "r") as f:
            if ext == ".json":
                data = json.load(f)
            elif ext in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        self.__build_graph(data)

    # private method
    def __build_graph(self, data: dict) -> None:
        # turns topology json data into graph format
        self.graph.clear()

        # initialise empty dicts for each node
        for node in data.get("nodes", []):
            self.graph[node] = {}

        for link in data.get("links", []):
            # get link parameters
            start, dest = link["from"], link["to"]
            latency, bandwidth = float(link.get("latency", 0)), float(link.get("bandwidth", 0))

            # set link parameters in graph
            self.graph[start][dest] = {"latency": latency, "bandwidth": bandwidth}

    def print_graph(self):
        for start in self.graph.keys():
            # self.graph[start], is the assigned dict
            for dest, params in self.graph[start].items():
                print(f"{start} -> {dest}: (latency: {params["latency"]}, bandwidth: {params["bandwidth"]})")
            print()


top = TopologyManager()
top.load_topology("configs/square-topology.json")
top.print_graph()
