# A bool member of a stored tuple reads back wrong, independently of any
# function call or parameter annotation. Found while fixing GitHub #5936, which
# therefore refuses to type bool tuple members: binding them would turn that
# issue's false proof into a false alarm. When this is fixed, promote to CORE
# and let param_annotations.cpp::constant_type_name type bool again.
pair: list = [(True, 5)]
first = pair[0]
assert first[1] == 5
