import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "src"
nb = json.load(open(path, encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    print(f"\n========== CELL {i} [{c['cell_type']}] ==========")
    if mode in ("src", "both"):
        print("".join(c["source"]))
    if mode in ("out", "both") and c["cell_type"] == "code":
        for j, o in enumerate(c.get("outputs", [])):
            t = o.get("output_type")
            if t == "stream":
                print(f"[stream {o.get('name')}]:")
                print("".join(o.get("text", [])))
            elif t in ("execute_result", "display_data"):
                data = o.get("data", {})
                if "text/plain" in data:
                    print(f"[{t} text/plain]:")
                    txt = data["text/plain"]
                    print("".join(txt) if isinstance(txt, list) else txt)
                if "image/png" in data:
                    print(f"[{t} image/png present]")
            elif t == "error":
                print(f"[error {o.get('ename')}: {o.get('evalue')}]")
