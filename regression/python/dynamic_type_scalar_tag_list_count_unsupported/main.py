# count/index/remove over a tagged element are not modelled: the untagged
# element-info path stamps the wrapper's static type hash, which never matches
# the element's runtime type_id, and count then proved `count(x) == 0`.
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
assert lst.count(x) == 0
