import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
lines = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream" and o.get("name") == "stdout":
            text = "".join(o.get("text", []))
            if "AR :" in text or "AR:" in text:
                lines.append(f"---- CELL {i} ----\n" + text)
with open("ar_samples.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote", "ar_samples.txt", "bytes:", sum(len(l) for l in lines))
