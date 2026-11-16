#pragma once

// std::min and std::max for two values. <char_traits.h> includes this, so
// <ios>, <iostream>, <string> and <vector> reach them as libstdc++ and libc++
// do (#7536); the rest of <algorithm> is not pulled in with them.

namespace std
{
// [alg.min.max]: only operator< is required, and a tie returns the first
// argument.
template <class T>
const T &max(const T &left, const T &right)
{
  return (left < right) ? right : left;
}

template <class T>
const T &min(const T &left, const T &right)
{
  return (right < left) ? right : left;
}
} // namespace std
