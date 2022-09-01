import os


def string2number(numb) -> str:
    if "-" in numb or "–" in numb:
        return "-"
    if "," in numb:
        numb = numb.split(",")
        return "".join(numb)
    else:
        return numb


def html2data():
    with open("excel.html", "r", encoding="utf-8") as rf:
        content = rf.readline()

    with open("data.csv", "w") as wf:
        rows = content.split("</tr>")[:-1]
        for row in rows:
            cells = row.split("</td>")[:-1]
            cells = [string2number(cell.split(">")[-1]) for cell in cells]
            wf.write(",".join(cells))
            wf.write("\n")       


def num(item):
    num_item = []
    for factor in item:
        if factor == "-":
            num_item.append(0)
        else:
            num_item.append(int(factor))
    return num_item

def carbonAve():
    with open("data.csv", "r") as rf:
        lines = rf.readlines()
        lines = [line.strip().split(",") for line in lines]
    with open("rf_data.csv", "w") as wf:
        wf.write("pred-AvgCO2,pred-EUI\n")
        for item in lines:
            item = num(item)
            eui = (item[-3]/(20*0.757)) / ((item[-3]+item[-2])/item[-1])
            wf.write("{},{}\n".format(item[-1], int(eui)))


if __name__ == "__main__":
    carbonAve()