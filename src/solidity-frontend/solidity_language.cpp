#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <clang/Frontend/ASTUnit.h>
CC_DIAGNOSTIC_POP()

#include <solidity-frontend/solidity_language.h>
#include <solidity-frontend/solidity_convert.h>
#include <clang-c-frontend/clang_c_main.h>
#include <clang-c-frontend/clang_c_adjust.h>
#include <clang-c-frontend/clang_c_convert.h>
#include <c2goto/cprover_library.h>
#include <clang-c-frontend/expr2c.h>
#include <util/c_link.h>

languaget *new_solidity_language()
{
  return new solidity_languaget;
}

std::string solidity_languaget::get_temp_file()
{
  // Create a temp file for clang-tool
  // needed to convert intrinsics
  auto p = boost::filesystem::temp_directory_path();
  if(!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
  {
    log_error("Can't find temporary directory (needed to convert intrinsics)");
    abort();
  }

  // Create temporary directory
  p += "/esbmc_solidity_temp";
  boost::filesystem::create_directory(p);
  if(!boost::filesystem::is_directory(p))
  {
    log_error(
      "Can't create temporary directory (needed to convert intrinsics)");
    abort();
  }

  // populate temp file
  std::ofstream f;
  p += "/temp_sol.c";
  f.open(p.string());
  f << temp_c_file();
  f.close();

  return p.string();
}

bool solidity_languaget::parse(const std::string &path)
{
  // prepare temp file
  temp_path = get_temp_file();

  // get AST nodes of ESBMC intrinsics and the dummy main
  // populate ASTs inherited from parent class
  clang_c_languaget::parse(temp_path);

  // Process AST json file
  std::ifstream ast_json_file_stream(path);
  std::string new_line, ast_json_content;

  while(getline(ast_json_file_stream, new_line))
  {
    if(new_line.find(".sol =======") != std::string::npos)
    {
      break;
    }
  }
  while(getline(ast_json_file_stream, new_line))
  {
    // file pointer continues from "=== *.sol ==="
    if(new_line.find(".sol =======") == std::string::npos)
    {
      ast_json_content = ast_json_content + new_line + "\n";
    }
    else
    {
      assert(!"Unsupported feature: found multiple contracts defined in a single .sol file");
    }
  }

  // parse explicitly
  ast_json = nlohmann::json::parse(ast_json_content);

  return false;
}

bool solidity_languaget::convert_intrinsics(contextt &context)
{
  clang_c_convertert converter(context, ASTs, "C");
  if(converter.convert())
    return true;
  return false;
}

bool solidity_languaget::typecheck(contextt &context, const std::string &module)
{
  contextt new_context;
  convert_intrinsics(
    new_context); // Add ESBMC and TACAS intrinsic symbols to the context
  log_progress("Done conversion of intrinsics...");

  solidity_convertert converter(
    new_context, ast_json, func_name, smart_contract);
  if(converter.convert()) // Add Solidity symbols to the context
    return true;

  clang_c_adjust adjuster(new_context);
  if(adjuster.adjust())
    return true;

  if(c_link(
       context,
       new_context,
       module)) // also populates language_uit::context
    return true;

  return false;
}

void solidity_languaget::show_parse(std::ostream &)
{
  assert(!"come back and continue - solidity_languaget::show_parse");
}

bool solidity_languaget::final(contextt &context)
{
  add_cprover_library(context);
  return clang_main(context);
}

std::string solidity_languaget::temp_c_file()
{
  // This function populates the temp file so that Clang has a compilation job.
  // Clang needs a job to convert the intrinsics.
  std::string content = R"(int main() { return 0; } )";
  return content;
}
