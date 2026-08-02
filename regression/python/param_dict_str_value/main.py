# Reading a string value from an unannotated dict (#5444): the parameter d has
# no compile-time element type, so the element is erased to void* and its
# runtime str type is recorded in item->type_id. extract_pyobject_value
# dispatches on that type_id to keep the stored char* pointer instead of
# overrunning it with an 8-byte void* dereference. This exercises the if-select
# migrated to IREP2 (if2tc/equality2tc).
def get_val(d):
    return d['k']


assert get_val({'k': 'hello'}) == 'hello'
