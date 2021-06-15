/*******************************************************************\

Module: Solidity AST module

\*******************************************************************/

#include <solidity-ast-frontend/solidity_ast_language.h>
#include <solidity-ast-frontend/solidity_convert.h>

languaget *new_solidity_ast_language()
{
  return new solidity_ast_languaget;
}

solidity_ast_languaget::solidity_ast_languaget()
{
  clang_c_module = new_clang_c_language();
}

solidity_ast_languaget::~solidity_ast_languaget()
{
  if(clang_c_module != nullptr)
    delete clang_c_module;
}

bool solidity_ast_languaget::parse(
  const std::string &path,
  message_handlert &message_handler)
{
  printf("sol_main_path: %s\n", sol_main_path.c_str());
  //assert(sol_main_path != ""); // we don't need a 'main' function if --function is used

  // get AST nodes of ESBMC intrinsics and the dummy main
  clang_c_module->parse(sol_main_path, message_handler); // populate clang_c_module's ASTs

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

  //print_json(ast_json);

  return false;
}

bool solidity_ast_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  contextt new_context;
  clang_c_module->convert_intrinsics(new_context);

  solidity_convertert converter(new_context, ast_json);

  if(converter.convert())
    return true;

  assert(!"come back and continue - solidity_ast_languaget::typecheck");
  return false;
}

void solidity_ast_languaget::show_parse(std::ostream &)
{
  assert(!"come back and continue - solidity_ast_languaget::show_parse");
}

bool solidity_ast_languaget::final(
  contextt &context,
  message_handlert &message_handler)
{
  assert(!"come back and continue - solidity_ast_languaget::final");
  return false;
}

bool solidity_ast_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  assert(!"come back and continue - solidity_ast_languaget::from_expr");
  return false;
}

bool solidity_ast_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  assert(!"come back and continue - solidity_ast_languaget::from_type");
  return false;
}

void solidity_ast_languaget::print_json(const nlohmann::json &json_in)
{
  printf("\n### json_content: ###\n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations
  printf("\n");
}
