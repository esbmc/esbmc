#include "holder.h"

// `tab.at(i).resize(inner)` is the first instantiation of
// `std::vector<bool>::resize(int)` in this build. The crash in
// adjust_cpp_member fired here when `main.cpp` was parsed first and
// completed `vector<bool>` without seeing this method.
holder::holder(int outer, int inner)
{
  tab.resize(outer);
  for (int i = 0; i < outer; ++i)
    tab.at(i).resize(inner);
}
