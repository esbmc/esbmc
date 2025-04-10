#include <cassert>

struct char_traits;
struct basic_string;
template <typename, typename = char_traits>
struct basic_istream;
template <typename>
struct basic_ostream;

template <typename _Traits>
struct basic_iostream : basic_istream<_Traits, char_traits>,
                        basic_ostream<_Traits>
{
};
template <typename _Alloc>
void getline(basic_istream<char, char_traits>, _Alloc);
template <>
void getline(basic_istream<char, char_traits>, basic_string);
struct basic_ios
{
  basic_ostream<char_traits> *_M_tie;
};
template <typename>
struct basic_ostream : basic_ios
{
};
template <typename, typename>
struct basic_istream : virtual basic_ios
{
};
template struct basic_iostream<char_traits>;

int main()
{
  basic_ios ios_instance;

  basic_ostream<char_traits> ostream_instance;

  basic_istream<char, char_traits> istream_instance;
  basic_iostream<char_traits> iostream_instance;

  return 0;
}