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
}

bool solidity_ast_languaget::parse(
  const std::string &path,
  message_handlert &message_handler)
{
    /* store AST text and json contents in stream objects to facilitate parsing and node visiting */
    printf("plaintext ast path: %s\n", plaintext_ast_path.c_str());
    assert(plaintext_ast_path != "");

    std::ifstream ast_text_file_stream(plaintext_ast_path), ast_json_file_stream(path);
    std::stringstream ast_text_stream;
    std::string new_line, sol_name, ast_json_content;

    printf("\n### ast_text_file stream processing:... \n");
    while (getline(ast_text_file_stream, new_line)) {
        printf("new_line: %s\n", new_line.c_str());
        ast_text_stream << new_line << "\n"; // store AST text in stream object
    }

    while (getline(ast_json_file_stream, new_line)) {
        if (new_line.find(".sol =======") != std::string::npos) {
            printf("found .sol ====== , breaking ...\n");
            sol_name = Sif::Utils::substr_by_edge(new_line, "======= ", " =======");
            break;
        }
    }

    printf("\n### ast_json_file_stream processing:... \n");
    while (getline(ast_json_file_stream, new_line)) { // file pointer continues from "=== *.sol ==="
        //printf("new_line: %s\n", new_line.c_str());
        if (new_line.find(".sol =======") == std::string::npos) {
            ast_json_content = ast_json_content + new_line + "\n";
        }
        else {
            assert(!"Unsupported feature: found multiple contracts defined in a single .sol file");
        }
    }
    printf("\n ### ast_json_content: ###");
    printf("%s", ast_json_content.c_str());
    printf("\n");

    std::string visitor_arg = "";
    if (ast_json_content != "") {
        ast_json = nlohmann::json::parse(ast_json_content); // parse explicitly
        //std::cout << "This is ast_json: " << ast_json.at("absolutePath") << std::endl;
        Sif::ASTAnalyser ast_analyser(ast_text_stream, ast_json, true, sol_name, visitor_arg);

        Sif::RootNodePtr root_node = ast_analyser.analyse();
        Sif::Indentation indentation;
        std::string new_source = root_node->source_code(indentation);
        //std::cout << sol_name << " " << ast_json.at("absolutePath") << std::endl;
        std::string output_file_name = "generated_contract.sol";
        if (output_file_name != "") {
            std::ofstream output_file_stream(output_file_name);
            output_file_stream << new_source;
            output_file_stream.close();
        } else {
            assert(!"should not be here");
        }
    }

    ast_json_file_stream.close();

    return false;
}

bool solidity_ast_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  contextt new_context;

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
