/*******************************************************************\

Module: Solidity AST module

\*******************************************************************/

#include <solidity-frontend/solidity_language.h>
#include <solidity-frontend/solidity_convert.h>

languaget *new_solidity_language(const messaget &msg)
{
  return new solidity_languaget(msg);
}

solidity_languaget::solidity_languaget(const messaget &msg) : languaget(msg)
{
  clang_c_module = new_clang_c_language(msg);
}

solidity_languaget::~solidity_languaget()
{
  if(clang_c_module != nullptr)
    delete clang_c_module;
}

bool solidity_languaget::parse(
  const std::string &path,
  const messaget &msg)
{
  printf("sol_main_path: %s\n", sol_main_path.c_str());
  assert(sol_main_path != ""); // We still need a 'main' function although --function is provided.

  // get AST nodes of ESBMC intrinsics and the dummy main
  clang_c_module->parse(sol_main_path, msg); // populate clang_c_module's ASTs

  // Process AST json file
  std::ifstream ast_json_file_stream(path);
  std::string new_line, sol_name, ast_json_content;

  printf("\n### ast_json_file_stream processing:... \n");
  while (getline(ast_json_file_stream, new_line)) {
    if (new_line.find(".sol =======") != std::string::npos) {
      printf("found .sol ====== , breaking ...\n");
      sol_name = Sif::Utils::substr_by_edge(new_line, "======= ", " =======");
      break;
    }
  }
  while (getline(ast_json_file_stream, new_line)) { // file pointer continues from "=== *.sol ==="
    //printf("new_line: %s\n", new_line.c_str());
    if (new_line.find(".sol =======") == std::string::npos)
    {
      //printf("  new_line: ");
      //printf("%s\n", new_line.c_str());
      ast_json_content = ast_json_content + new_line + "\n";
    }
    else
    {
      assert(!"Unsupported feature: found multiple contracts defined in a single .sol file");
    }
  }
  ast_json = nlohmann::json::parse(ast_json_content); // parse explicitly

  //printf("@@ This is ast_json: \n");
  //print_json(ast_json);

  return false;
}

bool solidity_languaget::typecheck(
  contextt &context,
  const std::string &module,
  const messaget &msg)
{
  contextt new_context(msg);
  clang_c_module->convert_intrinsics(new_context, msg); // Add ESBMC and TACAS intrinsic symbols to the context
  msg.progress("Done conversion of intrinsics...");

  solidity_convertert converter(new_context, ast_json, sol_func_path, msg);
  if(converter.convert()) // Add Solidity symbols to the context
    return true;

  assert(!"continue with adjuster ...");
  clang_c_adjust adjuster(new_context, msg);
  if(adjuster.adjust())
    return true;

  if(c_link(context, new_context, msg, module)) // also populates language_uit::context
    return true;

  return false;
}

void solidity_languaget::show_parse(std::ostream &)
{
  assert(!"come back and continue - solidity_languaget::show_parse");
}

bool solidity_languaget::final(
  contextt &context,
  const messaget &msg)
{
  add_cprover_library(context, msg);
  return clang_main(context, msg);
  //assert(!"come back and continue - solidity_languaget::final");
  return false;
}

bool solidity_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  assert(!"should not be here - Solidity frontend does not need this funciton");
  return false;
}

bool solidity_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  assert(!"should not be here - Solidity frontend does not need this funciton");
  return false;
}

void solidity_languaget::print_json(const nlohmann::json &json_in)
{
  printf("\n### json_content: ###\n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations
  printf("\n");
}
