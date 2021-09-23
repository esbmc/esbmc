#include <fstream>
#include <jimple-frontend/AST/jimple_file.h>
#include <jimple-frontend/jimple-language.h>
#include <util/message/format.h>


void jimple_languaget::show_parse(std::ostream &out)
{
  out << root.to_string();
}

bool jimple_languaget::parse(const std::string &path, const messaget &msg)
{
  msg.debug(fmt::format("Parsing: {}", path));
  try {
    root.load_file(path);
  }

  catch(std::exception &e)
  {
    msg.error(e.what());
    return true;
  }

  return false;
}
