import os
from random import shuffle


def string2number(numb) -> str:
    if "-" in numb or "–" in numb:
        return "-"
    if "," in numb:
        numb = numb.split(",")
        return "".join(numb)
    else:
        return numb


def html2data():
    """
    Transform html into csv file
    excel.html -> data.csv
    """
    with open("excel.html", "r", encoding="utf-8") as rf:
        content = rf.readline()

    with open("data.csv", "w") as wf:
        rows = content.split("</tr>")[:-1]
        for row in rows:
            cells = row.split("</td>")[:-1]
            cells = [string2number(cell.split(">")[-1]) for cell in cells]
            wf.write(",".join(cells))
            wf.write("\n")       


def num(item: list):
    """
    convert an item in string form into int form
    "-" --> 0
    str --> int
    """
    num_item = []
    for factor in item:
        if factor == "-":
            num_item.append(0)
        else:
            num_item.append(int(factor))
    return num_item


def carbonAve(file_name):
    """
    Calculate AvgCO2 and EUI according to source data
    file --> rf_file
    """
    with open(file_name, "r") as rf:
        lines = rf.readlines()
        lines = [line.strip().split(",") for line in lines]
    with open("rf_"+file_name, "w") as wf:
        wf.write("pred-AvgCO2,pred-EUI\n")
        for item in lines:
            item = num(item)
            eui = (item[-3]/(20*0.757)) / ((item[-3]+item[-2])/item[-1])
            wf.write("{},{}\n".format(item[-1], int(eui)))


def split_data():
    """
    Shuffle source data.
    Split into train-eval data and pred data
    data.csv -> train-eval.csv, pred.csv
    """
    with open("data.csv", "r") as rf:
        lines = rf.readlines()
    shuffle(lines)
    train_eval_size = int(0.8*len(lines))
    with open("train-eval.csv", "w") as wf:
        for i in range(train_eval_size):
            wf.write(lines[i])
    with open("test.csv", "w") as wf:
        for i in range(train_eval_size, len(lines)):
            wf.write(lines[i])

if __name__ == "__main__":
    carbonAve(file_name="test.csv")