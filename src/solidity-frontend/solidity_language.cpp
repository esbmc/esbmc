#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Frontend/ASTUnit.h>
CC_DIAGNOSTIC_POP()

#include <solidity-frontend/solidity_language.h>
#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/solidity_template.h>
#include <clang-cpp-frontend/clang_cpp_main.h>
#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <c2goto/cprover_library.h>
#include <util/c_link.h>
#include "filesystem.h"

languaget *new_solidity_language()
{
  return new solidity_languaget;
}

solidity_languaget::solidity_languaget()
{
  std::string fun = config.options.get_option("function");
  if (!fun.empty())
    func_name = fun;
  else
    func_name = "";

  std::string cnt = config.options.get_option("contract");
  if (!cnt.empty())
    contract_names = cnt;
  else
    contract_names = "";

  std::string sol = config.options.get_option("sol");
  if (sol.empty())
  {
    log_error("Please set the smart contract source file via --sol");
    abort();
  }
  contract_path = sol;
}

std::string solidity_languaget::get_temp_file()
{
  // Create a temp file for clang-tool
  // needed to convert intrinsics
  static std::once_flag flag;
  static std::string p;

  std::call_once(flag, [&]() {
    p = file_operations::create_tmp_dir("esbmc_solidity_temp-%%%%-%%%%-%%%%")
          .path();
    boost::filesystem::create_directories(p);
    p += "/libary.cpp";
    std::ofstream f(p);
    if (!f)
    {
      log_error(
        "Can't create temporary directory (needed to convert intrinsics)");
      abort();
    }
    f << temp_cpp_file();
  });

  return p;
}

bool solidity_languaget::parse(const std::string &path)
{
  /// For c
  // prepare temp file
  temp_path = get_temp_file();

  // get AST nodes of ESBMC intrinsics and the dummy main
  // populate ASTs inherited from parent class
  auto sol_lang = std::exchange(config.language, {language_idt::CPP, ""});
  if (clang_cpp_languaget::parse(temp_path))
    return true;

  /// For solidity
  config.language = std::move(sol_lang);

  // Process AST json file
  std::ifstream ast_json_file_stream(path);
  std::string new_line;
  std::vector<nlohmann::json> json_blocks;
  std::string current_json_block;

  // Skip the initial part until the first ".sol ======="
  while (getline(ast_json_file_stream, new_line))
  {
    if (new_line.find(".sol =======") != std::string::npos)
    {
      break;
    }
  }
  // Read and parse each JSON block separately
  while (getline(ast_json_file_stream, new_line))
  {
    if (new_line.find(".sol =======") != std::string::npos)
    {
      if (!current_json_block.empty())
      {
        json_blocks.push_back(nlohmann::json::parse(current_json_block));
        current_json_block.clear();
      }
    }
    else
    {
      current_json_block += new_line + "\n";
    }
  }

  // Parse the last JSON block
  if (!current_json_block.empty())
  {
    json_blocks.push_back(nlohmann::json::parse(current_json_block));
  }

  // Combine all parsed JSON blocks into one JSON array
  for (const auto &block : json_blocks)
  {
    src_ast_json_array.push_back(block);
  }
  return false;
}

bool solidity_languaget::convert_intrinsics(contextt &context)
{
  clang_cpp_convertert converter(context, AST, "C++");
  if (converter.convert())
    return true;

  return false;
}

bool solidity_languaget::typecheck(contextt &context, const std::string &module)
{
  contextt new_context;
  convert_intrinsics(
    new_context); // Add ESBMC and TACAS intrinsic symbols to the context

  solidity_convertert converter(
    new_context, src_ast_json_array, contract_names, func_name, contract_path);
  if (converter.convert()) // Add Solidity symbols to the context
    return true;

  // migrate from clang_c_adjust to clang_cpp_adjust
  // for the reason that we need clang_cpp_adjust::adjust_side_effect
  // to adjust the created temporary object
  // otherwise it would raise "unknown side effect: temporary_object"
  clang_cpp_adjust adjuster(new_context);
  if (adjuster.adjust())
    return true;

  if (c_link(
        context, new_context, module)) // also populates language_uit::context
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
  clang_cpp_maint c_main(context);
  if (c_main.clang_main())
    return true;

  // roll back
  config.language = {language_idt::SOLIDITY, ""};
  return false;
}

std::string solidity_languaget::temp_cpp_file()
{
  // This function populates the temp file so that Clang has a compilation job.
  // Clang needs a job to convert the intrinsics.
  std::string content =
    SolidityTemplate::sol_library + R"(int main() { return 0; })";
  return content;
}
