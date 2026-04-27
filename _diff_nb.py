import json, sys, io, difflib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def cells_to_lines(path):
    nb = json.load(open(path, encoding="utf-8"))
    lines = []
    for i, c in enumerate(nb["cells"]):
        lines.append(f"\n========== CELL {i} [{c['cell_type']}] ==========")
        lines.extend(("".join(c["source"]) ).splitlines())
        if c["cell_type"] == "code":
            for o in c.get("outputs", []):
                t = o.get("output_type")
                if t == "stream":
                    lines.append(f"[stream {o.get('name')}]")
                    lines.extend(("".join(o.get("text", []))).splitlines())
                elif t in ("execute_result", "display_data"):
                    data = o.get("data", {})
                    if "text/plain" in data:
                        lines.append(f"[{t} text/plain]")
                        txt = data["text/plain"]
                        lines.extend((("".join(txt)) if isinstance(txt, list) else txt).splitlines())
                    if "image/png" in data:
                        lines.append(f"[{t} image/png present]")
    return lines

a = cells_to_lines(sys.argv[1])
b = cells_to_lines(sys.argv[2])
for line in difflib.unified_diff(a, b, fromfile=sys.argv[1], tofile=sys.argv[2], lineterm=""):
    print(line)
