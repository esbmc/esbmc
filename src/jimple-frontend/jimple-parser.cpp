#include <fstream>
#include <nlohmann/json.hpp>
#include <jimple-frontend/AST/jimple_file.h>
#include <jimple-frontend/jimple-language.h>
#include <util/message/format.h>

using json = nlohmann::json;


bool jimple_languaget::parse(const std::string &path, const messaget &msg)
{
  msg.debug(fmt::format("Parsing: {}",path));

  // Read from JSON
  std::ifstream i(path);
  json j;
  i >> j;

  // Parse the Top Level Structure
  auto jfile = j.get<jimple_file>();
  jfile.dump();
  return true;
}
