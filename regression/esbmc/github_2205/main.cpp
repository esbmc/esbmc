#include <assert.h>

struct iterator
{
  struct
  {
    int second;
    int it_pos;
  } iterator_data_object;
};
struct map
{
  struct
  {
    int _value[2];
  } map_data_object;
};
int main()
{
  map mymap;
  iterator it;
  mymap.map_data_object._value[0] = 1337;
  mymap.map_data_object._value[1] = 4269;
  int *it_0 = mymap.map_data_object._value;
  it.iterator_data_object.second = it_0[it.iterator_data_object.it_pos];
  it_0[it.iterator_data_object.it_pos];
}
