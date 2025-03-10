def mutate_list(elem:str, elems : list[str] = []):
   elems.append(elem)
   return elems

assert(len(mutate_list("a")) == 1)
assert(len(mutate_list("a")) == 2)
assert(len(mutate_list("a")) == 3)
assert(len(mutate_list("a")) == 4)