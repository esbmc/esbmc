// github.com/esbmc/esbmc/issues/4245 — disengaged optional must trip value().
#include <optional>

int main()
{
  std::optional<int> o;
  return o.value();
}
