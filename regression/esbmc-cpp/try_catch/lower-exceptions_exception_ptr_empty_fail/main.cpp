// Rethrowing an empty exception_ptr calls std::terminate().
#include <exception>

int main()
{
  std::exception_ptr ep;
  std::rethrow_exception(ep);
  return 0;
}
