# bytes (immutable) remains modeled and must still verify after the bytearray
# diagnostic was added next to it in type_handler::get_typet.
b = bytes([65, 66, 67])
assert b[0] == 65
assert b[2] == 67
