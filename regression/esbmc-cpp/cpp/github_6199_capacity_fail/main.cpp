#include <string>

int main()
{
  // n >= STRING_CAPACITY (128) cannot be represented by the model's fixed
  // buffer, so the constructor reports the capacity limit rather than writing
  // out of bounds. This is a model limitation, not a claim about C++ semantics.
  const char *s = "abc";
  std::string big(s, 200);

  return (int)big.length();
}
