#pragma once

// std::min and std::max for two values. <char_traits.h> includes this, so
// <ios>, <iostream>, <string> and <vector> reach them as libstdc++ and libc++
// do (#7536); the rest of <algorithm> is not pulled in with them.

namespace std
{
template <class T>
const T &max(const T &left, const T &right)
{
  if (left > right)
    return left;
  else
    return right;
}

template <class T>
const T &min(const T &left, const T &right)
{
  if (left < right)
    return left;
  else
    return right;
}
} // namespace std
