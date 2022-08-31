def string2number(numb) -> str:
    if "-" in numb or "–" in numb:
        return "-"
    if "," in numb:
        numb = numb.split(",")
        return "".join(numb)
    else:
        return numb


with open("excel.html", "r", encoding="utf-8") as rf:
    content = rf.readline()

with open("data.csv", "w") as wf:
    rows = content.split("</tr>")[:-1]
    for row in rows:
        cells = row.split("</td>")[:-1]
        cells = [string2number(cell.split(">")[-1]) for cell in cells]
        wf.write(",".join(cells))
        wf.write("\n")       
