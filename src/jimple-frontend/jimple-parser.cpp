#include <fstream>
#include <jimple-frontend/AST/jimple_file.h>
#include <jimple-frontend/jimple-language.h>

void jimple_languaget::show_parse(std::ostream &out)
{
  out << root.to_string();
}

bool jimple_languaget::parse(const std::string &path)
{
  log_debug("jimple", "Parsing: {}", path);
  try
  {
    root.load_file(path);
  }

  catch (std::exception &e)
  {
    log_error("{}", e.what());
    return true;
  }

  return false;
}
