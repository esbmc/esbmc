/// \file solidity_language.cpp
/// \brief Implementation of the Solidity language frontend.
///
/// Handles solc invocation to obtain the JSON AST, multi-contract resolution,
/// import flattening, and the parse/typecheck/final pipeline that converts
/// Solidity source files into ESBMC's GOTO program representation.

#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Frontend/ASTUnit.h>
CC_DIAGNOSTIC_POP()

#include <solidity-frontend/solidity_language.h>
#include <solidity-frontend/solidity_convert.h>
#include <clang-cpp-frontend/clang_cpp_main.h>
#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <c2goto/cprover_library.h>
#include <util/c_link.h>
#include "filesystem.h"
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <regex>

// Use boost::process v1 on macOS or when Boost >= 1.87
#if defined(__APPLE__) || (BOOST_VERSION >= 108700)
#  include <boost/process/v1.hpp>
namespace bp = boost::process::v1;
#else
#  include <boost/process.hpp>
namespace bp = boost::process;
#endif

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

  focus_func_name = config.options.get_option("focus-function");

  // contract_path may be set explicitly via --sol, or derived later in parse()
  // when the user passes a .sol file directly as a positional argument.
  std::string sol = config.options.get_option("sol");
  if (!sol.empty())
    contract_path = sol;
}

std::string solidity_languaget::get_temp_file()
{
  // Create a minimal temp file for clang-tool to parse ESBMC intrinsic symbols.
  // Only includes standard headers (for nondet, assert, etc.) and a dummy main.
  // Solidity operational models are loaded separately from c2goto (sol64).
  static std::once_flag flag;
  static std::string p;

  std::call_once(flag, [&]() {
    p = file_operations::create_tmp_dir("esbmc_solidity_temp-%%%%-%%%%-%%%%")
          .path();
    boost::filesystem::create_directories(p);
    p += "/intrinsics.cpp";
    std::ofstream f(p);
    if (!f)
    {
      log_error(
        "Can't create temporary directory (needed to convert intrinsics)");
      abort();
    }
    f << R"(
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>
int main() { return 0; }
)";
  });

  return p;
}

std::string solidity_languaget::find_solc() const
{
  // Priority: --solc-bin > $SOLC > solc in $PATH
  std::string bin = config.options.get_option("solc-bin");
  if (!bin.empty())
  {
    if (!boost::filesystem::exists(bin))
    {
      log_error("solc binary not found at: {}", bin);
      return "";
    }
    return bin;
  }

  const char *env = std::getenv("SOLC");
  if (env && env[0] != '\0')
  {
    if (!boost::filesystem::exists(env))
    {
      log_error("$SOLC points to non-existent path: {}", env);
      return "";
    }
    return env;
  }

  // Search $PATH for solc via boost::process (portable replacement for `which`)
  boost::filesystem::path found = bp::search_path("solc");
  if (!found.empty())
    return found.string();

  return "";
}

std::string solidity_languaget::get_solc_version(const std::string &solc) const
{
  // Run `solc --version`; merge stdout+stderr by reading both streams.
  bp::ipstream out_stream;
  bp::ipstream err_stream;
  std::string output;
  try
  {
    bp::child proc(
      solc, "--version", bp::std_out > out_stream, bp::std_err > err_stream);

    std::string line;
    while (out_stream && std::getline(out_stream, line))
      output += line + "\n";
    while (err_stream && std::getline(err_stream, line))
      output += line + "\n";

    proc.wait();
  }
  catch (const std::exception &)
  {
    return "unknown";
  }

  // Extract version number (e.g. "0.8.28+commit...")
  std::regex ver_re(R"((\d+\.\d+\.\d+))");
  std::smatch match;
  if (std::regex_search(output, match, ver_re))
    return match[1].str();

  return "unknown";
}

bool solidity_languaget::invoke_solc(
  const std::string &sol_path,
  std::string &solast_path)
{
  std::string solc = find_solc();
  if (solc.empty())
  {
    log_error(
      "solc not found. Install solc or specify its path with --solc-bin "
      "or $SOLC environment variable.\n"
      "Alternatively, generate the AST manually:\n"
      "  solc --ast-compact-json {} > {}.solast\n"
      "  esbmc --sol {} {}.solast",
      sol_path,
      sol_path,
      sol_path,
      sol_path);
    return true;
  }

  std::string version = get_solc_version(solc);
  log_status("Compiling Solidity AST using: {} (v{})", solc, version);

  // Create temp directory for the .solast output
  std::string tmp_dir =
    file_operations::create_tmp_dir("esbmc_solast-%%%%-%%%%-%%%%").path();
  boost::filesystem::create_directories(tmp_dir);
  solast_path = tmp_dir + "/output.solast";

  std::string cmd =
    solc + " --ast-compact-json " + sol_path + " > " + solast_path + " 2>&1";
  int ret = std::system(cmd.c_str());
  if (ret != 0)
  {
    // Read and display solc error output
    std::ifstream err_file(solast_path);
    std::string err_output(
      (std::istreambuf_iterator<char>(err_file)),
      std::istreambuf_iterator<char>());
    log_error("solc compilation failed:\n{}", err_output);
    return true;
  }

  return false;
}

bool solidity_languaget::parse_solast(const std::string &path)
{
  std::ifstream ast_json_file_stream(path);
  if (!ast_json_file_stream.is_open())
  {
    log_error("Cannot open AST file: {}", path);
    return true;
  }

  std::string new_line;
  std::vector<nlohmann::json> json_blocks;
  std::string current_json_block;

  // Skip the initial part until the first ".sol ======="
  while (getline(ast_json_file_stream, new_line))
  {
    if (new_line.find(".sol =======") != std::string::npos)
      break;
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
    json_blocks.push_back(nlohmann::json::parse(current_json_block));

  // Combine all parsed JSON blocks into one JSON array
  for (const auto &block : json_blocks)
    src_ast_json_array.push_back(block);

  return false;
}

bool solidity_languaget::parse(const std::string &path)
{
  // Phase 1: Parse a minimal C++ file through Clang to get ESBMC intrinsic
  // symbols (nondet_bool, nondet_uint, __ESBMC_assert, etc.)
  temp_path = get_temp_file();
  auto sol_lang = std::exchange(config.language, {language_idt::CPP, ""});
  if (clang_cpp_languaget::parse(temp_path))
    return true;
  config.language = std::move(sol_lang);

  // Phase 2: Determine whether input is .sol (needs solc) or .solast (direct)
  std::string solast_path;
  bool is_sol_source =
    (path.size() >= 4 && path.substr(path.size() - 4) == ".sol");

  if (is_sol_source)
  {
    // Set contract_path to the .sol source if not already set via --sol
    if (contract_path.empty())
      contract_path = path;

    // Auto-invoke solc to generate AST
    if (invoke_solc(path, solast_path))
      return true;
  }
  else
  {
    // Input is .solast — use directly
    solast_path = path;

    if (contract_path.empty())
    {
      log_error(
        "When passing a .solast file directly, please also specify "
        "the .sol source via --sol");
      return true;
    }
  }

  // Phase 3: Parse the Solidity AST JSON
  return parse_solast(solast_path);
}

bool solidity_languaget::convert_intrinsics(contextt &context)
{
  clang_cpp_convertert converter(context, AST, "C++");
  if (converter.convert())
    return true;

  // The C++ contract-intrinsic header defines __ESBMC_return_value as a
  // C++ definition. The Solidity operational-model goto binary (sol64)
  // also carries it as a C-mode symbol. Drop the C++ version here so the
  // phase-2 add_cprover_library merge does not see a duplicate.
  context.erase_symbol("c:@__ESBMC_return_value");

  return false;
}

bool solidity_languaget::typecheck(contextt &context, const std::string &module)
{
  contextt new_context;

  // Phase 1: Convert ESBMC intrinsic symbols (nondet_bool, nondet_uint,
  // __ESBMC_assert, etc.) from Clang AST into the context.
  convert_intrinsics(new_context);

  // Phase 2: Load Solidity operational models from the separate sol64 goto
  // binary (builtins, mapping, array, bytes, string, address, units, misc).
  add_cprover_library(new_context, this);

  // Record which symbols came from phases 1+2 (before converter adds its own)
  std::unordered_set<std::string> lib_symbols;
  new_context.foreach_operand(
    [&lib_symbols](const symbolt &s) { lib_symbols.insert(s.id.as_string()); });

  // Phase 3: Convert Solidity AST to ESBMC IR
  solidity_convertert converter(
    new_context,
    src_ast_json_array,
    contract_names,
    func_name,
    contract_path,
    focus_func_name);
  if (converter.convert())
    return true;

  // Phase 4: Adjust converter-generated code. Save and restore sol64 function
  // bodies because clang_cpp_adjust would corrupt them (they are already
  // adjusted by c2goto's clang_c_adjust).
  std::unordered_map<std::string, exprt> saved_values;
  new_context.Foreach_operand([&](symbolt &s) {
    if (lib_symbols.count(s.id.as_string()) && s.value.is_not_nil())
      saved_values[s.id.as_string()] = s.value;
  });

  clang_cpp_adjust adjuster(new_context);
  if (adjuster.adjust())
    return true;

  // Restore pre-adjusted function bodies from intrinsics and sol64
  for (auto &[id, val] : saved_values)
  {
    symbolt *s = new_context.find_symbol(id);
    if (s)
      s->value = std::move(val);
  }

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
