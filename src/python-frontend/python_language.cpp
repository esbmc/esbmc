#include "python-frontend/python_language.h"
#include "python-frontend/python_converter.h"
#include "util/message.h"

#include <cstdlib>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>

namespace bp = boost::process;
namespace fs = boost::filesystem;

languaget *new_python_language()
{
  return new python_languaget;
}

bool python_languaget::parse(const std::string &path)
{
  log_debug("python", "Parsing: {}", path);

  fs::path script(path);
  if(!fs::exists(script))
    return true;

  // Execute python script to generate json file from AST
  std::string cmd("python3 src/python-frontend/astgen.py ");
  cmd += path;
  // Create a child process to execute Python
  bp::child process(cmd);

  // Wait for execution
  process.wait();

  if(process.exit_code())
  {
    // Python execution failed
    return true;
  }

  return false;
}

bool python_languaget::final(contextt & /*context*/)
{
  return false;
}

bool python_languaget::typecheck(
  contextt &context,
  const std::string & /*module*/)
{
  python_converter converter(context);
  return converter.convert();
}

void python_languaget::show_parse(std::ostream & /*out*/)
{
}

bool python_languaget::from_expr(
  const exprt & /*expr*/,
  std::string & /*code*/,
  const namespacet & /*ns*/)
{
  assert(!"Not implemented yet");
  return false;
}

bool python_languaget::from_type(
  const typet & /*type*/,
  std::string & /*code*/,
  const namespacet & /*ns*/)
{
  assert(!"Not implemented yet");
  return false;
}
