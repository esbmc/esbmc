#include <vector>

int main()
{
  // A heap-allocated vector that is never deleted genuinely leaks: the vector
  // destructor frees the internal buffer, but nothing frees the vector object
  // itself. --memory-leak-check must still report this.
  std::vector<int> *v = new std::vector<int>();
  v->push_back(1);

  return (*v)[0];
}
