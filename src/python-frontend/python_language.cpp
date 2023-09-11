#include <python-frontend/python_language.h>

languaget *new_python_language()
{
  return new python_languaget;
}

bool python_languaget::parse(const std::string & /*path*/)
{
  return true;
}

bool python_languaget::final(contextt & /*context*/)
{
  return true;
}

bool python_languaget::typecheck(
  contextt & /*context*/,
  const std::string & /*module*/)
{
  return true;
}

void python_languaget::show_parse(std::ostream & /*out*/)
{
}

bool python_languaget::from_expr(
  const exprt &/*expr*/,
  std::string &/*code*/,
  const namespacet &/*ns*/)
{
  assert(!"Not implemented yet");
  return false;
}

bool python_languaget::from_type(
  const typet &/*type*/,
  std::string &/*code*/,
  const namespacet &/*ns*/)
{
  assert(!"Not implemented yet");
  return false;
}
