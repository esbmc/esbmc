#include <string>

int main()
{
  // One past what the model can represent (STRING_CAPACITY == 128, and the
  // terminator needs str[n]). Relaxing the bound to n <= STRING_CAPACITY would
  // admit this and write out of bounds, so the capacity report is the contract.
  std::string s(128, 'z');

  return 0;
}
