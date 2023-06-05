import sys
import networkx as nx
import json


def convert(in_file, out_file):
    nodes = {}
    edges = {}

    with open(in_file, "r", encoding="utf-8") as jsonl:
        for line in jsonl.readlines():
            json_line = json.loads(line)
            for _, ed in json_line.items():
                if ed is not None:
                    id = ed.get('id')  # Get unique id
                    label = ed.get("label") if "label" in ed else ed.get("labels", [None])[0]
                    if label is None:
                        print(f"ERROR: Label not found for element: {id}")
                    props = ed.get("properties", {})
                    type = ed.get("type", None)
                    if type == "node":
                        props["label"] = label
                        props["node_type"] = label
                        props[f"{label}_label"] = label
                        nodes[id] = props
                    elif type == "relationship":
                        start = ed["start"]
                        end = ed["end"]
                        props["edge_type"] = label
                        edges[(start, end)] = props
                    else:
                        print(f"ERROR: Unrecognized type: {type}")

    G = nx.DiGraph()
    for identifier, props in nodes.items():
        G.add_node(identifier, **props)
    for (start, end), props in edges.items():
        G.add_edge(start, end, **props)

    nx.write_graphml(G, out_file)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        print(f"Usage: python json2graphml.py in_file [out_file]")
        exit()
    in_file = arguments[0]
    out_file = in_file.replace(".json", ".graphml")
    if len(arguments) >= 2:
        out_file = arguments[1]
    convert(in_file, out_file)
    print(f"Done, output file saved to: {out_file}")


