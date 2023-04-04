import re

name = 'configs/config14.json'

a = re.findall(r"\bc\w*\d", name)
print(a)