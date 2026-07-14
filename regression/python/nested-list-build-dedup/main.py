# In-function (@F scope) 2D list literal: the converter used to build the
# literal twice (once as a discarded type probe). Reads must still see the
# correct values after that dead duplicate is elided (#5121).
def main():
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    s = 0
    for i in range(3):
        for j in range(3):
            s = s + a[i][j] * b[i][j]
    assert s == 165

main()
