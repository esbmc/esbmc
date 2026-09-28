#include <cassert>

// `&row`, where row is a row of a 2-D array, has type `S (*)[2]`: taking the
// address of an array-typed element is not the array-to-pointer decay, and
// decaying it anyway makes the range-for's end pointer one *element* past the
// start rather than one row, so the second row read is out of bounds.
struct S
{
  int x;
};

int main()
{
  S cases[][2] = {{{1}, {2}}};

  for (auto &row : cases)
  {
    assert(row[0].x == 1);
    assert(row[1].x == 3);
  }
}
