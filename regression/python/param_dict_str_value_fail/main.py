# Companion to param_dict_str_value: the string value read back from the
# unannotated dict is 'hello', not 'world', so this assertion must fail. Pins
# that the type_id-dispatched if-select returns the correct stored string.
def get_val(d):
    return d['k']


assert get_val({'k': 'hello'}) == 'world'
