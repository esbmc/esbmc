from pathlib import Path

for path in Path('./').rglob('test.desc'):
    print(path)
    f = open(path,'r')
    lines = f.readlines()[:-1]
    lines.append("<item_10_mode>KNOWNBUG</item_10_mode>" + "\n")
    lines.append("</test-case>")
    f.close()
    f = open(path,'w')
    f.writelines(lines)
