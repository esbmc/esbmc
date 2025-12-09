# Scenario 1
data_arr = b'Hello'
#b = bytes([72, 101, 108, 108, 111])

assert data_arr[0] == 72
assert data_arr[1] == 101
assert data_arr[2] == 108
assert data_arr[3] == 108
assert data_arr[4] == 111

# Scenario 2
data_arr_2 = b'World'
#b2 = bytes([87, 111, 114, 108, 100])

assert data_arr_2[0] == 87
assert data_arr_2[1] == 111
assert data_arr_2[2] == 114
assert data_arr_2[3] == 108
assert data_arr_2[4] == 100

# Scenario 3
data_arr_3 = b'Python'
#b3 = bytes([80, 121, 116, 104, 111, 110])

assert data_arr_3[0] == 80
assert data_arr_3[1] == 121
assert data_arr_3[2] == 116
assert data_arr_3[3] == 104
assert data_arr_3[4] == 111
assert data_arr_3[5] == 110

# Additional assertions for scenario 3
assert len(data_arr_3) == 7
assert data_arr_3[-1] == 110

