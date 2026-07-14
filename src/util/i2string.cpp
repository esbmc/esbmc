#define USE_SPRINTF

#ifdef USE_SPRINTF
#  include <cstdio>
#  include <cstring>
#  include <util/i2string.h>
#else
#  include <sstream>
#  include <util/i2string.h>
#endif

std::string i2string(int i)
{
#ifdef USE_SPRINTF
  char buffer[100];
  snprintf(buffer, sizeof(buffer), "%d", i);
  return buffer;
#else
  std::ostringstream strInt;

  strInt << i;
  std::string str = strInt.str();

  return str;
#endif
}

std::string i2string(signed long int i)
{
#ifdef USE_SPRINTF
  char buffer[100];
  snprintf(buffer, sizeof(buffer), "%ld", i);
  return buffer;
#else
  std::ostringstream strInt;

  strInt << i;
  std::string str = strInt.str();

  return str;
#endif
}

std::string i2string(unsigned i)
{
#ifdef USE_SPRINTF
  char buffer[100];
  snprintf(buffer, sizeof(buffer), "%u", i);
  return buffer;
#else
  std::ostringstream strInt;

  strInt << i;
  std::string str = strInt.str();

  return str;
#endif
}

std::string i2string(unsigned long int i)
{
#ifdef USE_SPRINTF
  char buffer[100];
  snprintf(buffer, sizeof(buffer), "%lu", i);
  return buffer;
#else
  std::ostringstream strInt;

  strInt << i;
  std::string str = strInt.str();

  return str;
#endif
}
