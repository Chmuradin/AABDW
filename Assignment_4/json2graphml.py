import sys
import networkx as nx
import json


def convert(in_file, out_file, full=False):
    elements = []
    with open(in_file, "r", encoding="utf-8") as jsonl:
        if not full:
            for line in jsonl.readlines():
                json_line = json.loads(line)
                for _, ed in json_line.items():
                    if ed.get("type") in ["path"]:
                        for node in ed.get("nodes", []):
                            elements.append(node)
                        for rel in ed.get("relationships", []):
                            elements.append(rel)
                    elif ed.get("type") in ["node", "relationship"]:
                        elements.append(ed)
        else:
            elements = [ed for ed in json.load(jsonl)]

    nodes = {}
    edges = {}

    for element in elements:
        if "labels" in element:
            label = element["labels"][0]
        elif "label" in element:
            label = element["label"]
        else:
            print(f"ERROR: Label not found for element: {element.get('id')}")
        props = element.get("properties", {})
        type = element.get("type", None)
        if type == "node":
            props["label"] = props.get("id")
            props["node_type"] = label
            props[f"{label}_label"] = props.get("id")
            nodes[props.get("id")] = props
        elif type == "relationship":
            start = element["start"]
            end = element["end"]
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
        print(f"Usage: python json2graphml.py in_file [out_file] [-f]")
        exit()
    in_file = arguments[0]
    full = "-f" in arguments
    out_file = in_file.replace(".json", ".graphml")
    if len(arguments) >= 2 and arguments[1] != "-f":
        out_file = arguments[1]
    elif len(arguments) >= 3 and arguments[2] != "-f":
        out_file = arguments[2]
    convert(in_file, out_file, full)
    print(f"Done, output file saved to: {out_file}")
