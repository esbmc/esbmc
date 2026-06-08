// An exception type deriving from std::exception is caught by a std::exception&
// base handler: the throw carries the flattened base chain (derived +
// std::exception), which the registry ingests, so the base catch's guard
// matches.
#include <exception>

struct MyError : std::exception
{
  const char *what() const noexcept override
  {
    return "my error";
  }
};

int main()
{
  int caught = 0;
  try
  {
    throw MyError();
  }
  catch (const std::exception &)
  {
    caught = 1;
  }
  __ESBMC_assert(caught == 1, "caught by std::exception base");
  return 0;
}
