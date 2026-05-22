#include <python-frontend/string/char_utils.h>
#include <python-frontend/complex_handler.h>
#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/convert_float_literal.h>
#include <python-frontend/function_call/builder.h>
#include <python-frontend/python_consteval.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/module_locator.h>
#include <python-frontend/python_annotation.h>
#include <python-frontend/python_class_builder.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_lambda.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_typechecking.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/encoding.h>
#include <util/expr_util.h>
#include <util/irep.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/symbolic_types.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <boost/filesystem.hpp>

using namespace json_utils;
namespace fs = boost::filesystem;

std::string
python_converter::get_op(const std::string &op, const typet &type) const
{
  return python_frontend::map_operator(op, type);
}

python_converter::python_converter(
  contextt &_context,
  const nlohmann::json *ast,
  const global_scope &gs)
  : symbol_table_(_context),
    ast_json(ast),
    global_scope_(gs),
    type_handler_(*this),
    string_builder_(new string_builder(*this, &string_handler_)),
    sym_generator_("python_converter::"),
    ns(_context),
    current_func_name_(""),
    current_class_name_(""),
    current_block(nullptr),
    current_lhs(nullptr),
    string_handler_(*this, symbol_table_, type_handler_, string_builder_),
    math_handler_(*this, symbol_table_, type_handler_),
    complex_handler_(*this, symbol_table_, type_handler_),
    tuple_handler_(new tuple_handler(*this, type_handler_)),
    dict_handler_(new python_dict_handler(*this, symbol_table_, type_handler_)),
    typechecker_(new python_typechecking(*this)),
    lambda_handler_(new python_lambda(*this, _context, type_handler_)),
    exception_handler_(new python_exception_handler(*this, type_handler_))
{
}

python_converter::~python_converter()
{
  delete string_builder_;
  delete tuple_handler_;
  delete dict_handler_;
  delete typechecker_;
  delete lambda_handler_;
  delete exception_handler_;
}

python_typechecking &python_converter::get_typechecker()
{
  return *typechecker_;
}

const python_typechecking &python_converter::get_typechecker() const
{
  return *typechecker_;
}

string_builder &python_converter::get_string_builder()
{
  if (!string_builder_)
  {
    string_builder_ = new string_builder(*this, &string_handler_);
    string_handler_.set_string_builder(string_builder_);
  }
  return *string_builder_;
}

static void add_global_static_variable(
  contextt &ctx,
  const typet t,
  const std::string &name)
{
  std::string id = "c:@" + name;
  symbolt symbol;
  symbol.mode = "C";
  symbol.get_type() = std::move(t);
  symbol.name = name;
  symbol.id = id;

  symbol.lvalue = true;
  symbol.static_lifetime = true;
  symbol.is_extern = false;
  symbol.file_local = false;
  symbol.get_value() = gen_zero(t, true);
  symbol.get_value().zero_initializer(true);

  symbolt *added_symbol = ctx.move_symbol_to_context(symbol);
  assert(added_symbol);
}

void python_converter::load_c_intrisics(code_blockt &)
{
  // Add symbols required by the C models
  // __ESBMC_rounding_mode is pulled in indirectly via fesetround in cprover_library.cpp

  auto type1 = array_typet(bool_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type1, "__ESBMC_alloc");
  add_global_static_variable(symbol_table_, type1, "__ESBMC_is_dynamic");

  auto type2 = array_typet(size_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type2, "__ESBMC_alloc_size");
}

///  Creates ``__name__`` and ``__file__``; ``__doc__`` / ``__package__``
///  remain unsupported.
void python_converter::create_builtin_symbols()
{
  const std::string module_name =
    current_python_file.substr(0, current_python_file.find_last_of("."));

  locationt location;
  location.set_file(current_python_file.c_str());
  location.set_line(1);

  auto add_string_builtin =
    [&](const std::string &builtin_name, const std::string &value) {
      symbol_id sid(current_python_file, "", "");
      sid.set_object(builtin_name);

      typet string_type =
        type_handler_.build_array(char_type(), value.size() + 1);

      symbolt sym = create_symbol(
        module_name, builtin_name, sid.to_string(), location, string_type);

      sym.lvalue = true;
      sym.static_lifetime = true;
      sym.is_extern = false;
      sym.file_local = false;

      exprt value_expr = gen_zero(string_type);
      const typet &char_type_ref = string_type.subtype();
      for (size_t i = 0; i < value.size(); ++i)
      {
        uint8_t ch = value[i];
        value_expr.operands().at(i) = constant_exprt(
          integer2binary(BigInt(ch), bv_width(char_type_ref)),
          integer2string(BigInt(ch)),
          char_type_ref);
      }
      value_expr.operands().at(value.size()) = constant_exprt(
        integer2binary(BigInt(0), bv_width(char_type_ref)),
        integer2string(BigInt(0)),
        char_type_ref);
      sym.get_value() = value_expr;

      symbol_table_.add(sym);
    };

  // Determine the value of __name__: "__main__" for the main module, else the
  // module's basename without extension.
  std::string name_value;
  if (current_python_file == main_python_file)
    name_value = "__main__";
  else
  {
    size_t last_slash = current_python_file.find_last_of("/\\");
    size_t last_dot = current_python_file.find_last_of(".");
    if (
      last_slash != std::string::npos && last_dot != std::string::npos &&
      last_dot > last_slash)
    {
      name_value =
        current_python_file.substr(last_slash + 1, last_dot - last_slash - 1);
    }
    else if (last_dot != std::string::npos)
      name_value = current_python_file.substr(0, last_dot);
    else
      name_value = current_python_file;
  }

  add_string_builtin("__name__", name_value);

  // __file__ mirrors CPython: the (absolute) path of the source file. We use
  // the canonical path the frontend already tracks via current_python_file.
  add_string_builtin("__file__", current_python_file);
}

bool python_converter::import_module_into_block(
  const nlohmann::json &import_node,
  module_locator &locator,
  code_blockt &block)
{
  const std::string &module_name = (import_node["_type"] == "ImportFrom")
                                     ? import_node["module"]
                                     : import_node["names"][0]["name"];

  if (imported_modules.find(module_name) != imported_modules.end())
    return true;

  // pre_collect_module_asts populates the pool before any import runs.
  // A miss here means the same module_locator could not open the file.
  auto pooled = module_ast_pool_.find(module_name);
  if (pooled == module_ast_pool_.end())
    return false;
  nlohmann::json &nested_module_json = pooled->second;

  current_python_file = nested_module_json["filename"].get<std::string>();
  imported_modules.emplace(module_name, current_python_file);

  // Process nested imports first.
  process_module_imports(nested_module_json, locator, block);

  // Then process this module's definitions.
  create_builtin_symbols();
  python_annotation<nlohmann::json> imported_annotator(
    nested_module_json, const_cast<global_scope &>(global_scope_));
  // Expose every other reachable module's AST as an extra subscript
  // inference source so that multi-axis subscript usages on instances of
  // this module's classes (e.g. `t[i:j, k:l]` where `Tile` is defined
  // here) are visible no matter which module they appear in. The
  // entry-point AST closes the 2-file form (GitHub #4545); intermediate
  // modules close the 3+-file transitive form (GitHub #4554).
  imported_annotator.add_extra_subscript_inference_source(*ast_json);
  for (auto &entry : module_ast_pool_)
    if (&entry.second != &nested_module_json)
      imported_annotator.add_extra_subscript_inference_source(entry.second);
  imported_annotator.add_type_annotation();

  exprt imported_code = with_ast(&nested_module_json, [&]() {
    return get_block(nested_module_json["body"]);
  });

  convert_expression_to_code(imported_code);

  // Add imported module code.
  block.copy_to_operands(imported_code);
  return true;
}

void python_converter::process_module_imports(
  const nlohmann::json &module_ast,
  module_locator &locator,
  code_blockt &block)
{
  // Process imports in this module first (depth-first)
  for (const auto &elem : module_ast["body"])
  {
    if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
    {
      std::string saved_file = current_python_file;
      import_module_into_block(elem, locator, block);
      current_python_file = saved_file;
    }
  }
}

void python_converter::pre_collect_module_asts(
  const nlohmann::json &module_ast,
  module_locator &locator)
{
  auto try_collect = [&](const nlohmann::json &node) {
    if (node["_type"] != "ImportFrom" && node["_type"] != "Import")
      return;
    if (node.value("module_not_found", false))
      return;
    const std::string module_name = (node["_type"] == "ImportFrom")
                                      ? node["module"]
                                      : node["names"][0]["name"];
    if (module_ast_pool_.count(module_name))
      return;
    std::ifstream f = locator.open_module_file(module_name);
    if (!f.is_open())
      return;
    nlohmann::json parsed;
    try
    {
      f >> parsed;
    }
    catch (const nlohmann::json::exception &)
    {
      // Treat a malformed AST as an unresolvable import; the subsequent
      // import_module_into_block lookup will return false the same way.
      return;
    }
    auto [it, _] = module_ast_pool_.emplace(module_name, std::move(parsed));
    pre_collect_module_asts(it->second, locator);
  };

  if (!module_ast.contains("body") || !module_ast["body"].is_array())
    return;

  for (const auto &elem : module_ast["body"])
  {
    try_collect(elem);
    if (
      elem["_type"] == "FunctionDef" && elem.contains("body") &&
      elem["body"].is_array())
      for (const auto &stmt : elem["body"])
        try_collect(stmt);
  }
}

void python_converter::convert()
{
  main_python_file = (*ast_json)["filename"].get<std::string>();
  current_python_file = main_python_file;

  // Create built-in symbols for main module (__name__ = "__main__")
  create_builtin_symbols();

  // Block to accumulate model library code
  code_blockt models_block;

  if (!config.options.get_bool_option("no-library"))
  {
    // Load operational models
    const std::string &ast_output_dir =
      (*ast_json)["ast_output_dir"].get<std::string>();
    std::list<std::string> model_files = {
      "builtins",
      "range",
      "int",
      "consensus",
      "random",
      "exceptions",
      "datetime",
      "nondet"};
    std::list<std::string> model_folders = {"os", "numpy"};

    for (const auto &folder : model_folders)
    {
      append_models_from_directory(model_files, ast_output_dir + "/" + folder);
    }

    is_loading_models = true;

    for (const auto &file : model_files)
    {
      std::stringstream model_path;
      model_path << ast_output_dir << "/" << file << ".json";

      std::ifstream model_file(model_path.str());
      nlohmann::json model_json;
      if (!model_file.is_open())
      {
        // parser.py exited before producing this model — the user's
        // program almost certainly hit an unresolvable import that
        // aborted the AST generation pipeline (issue #2012). Surface
        // a structured error instead of letting the downstream
        // ``>> model_json`` throw an uncaught nlohmann parse_error.
        log_error(
          "Python frontend: missing operational-model AST '{}'. "
          "This usually means parser.py exited before generating it; "
          "check the parser output above for the underlying error.",
          model_path.str());
        exit(1);
      }
      try
      {
        model_file >> model_json;
      }
      catch (const nlohmann::json::exception &e)
      {
        log_error(
          "Python frontend: failed to parse operational-model AST "
          "'{}': {}.",
          model_path.str(),
          e.what());
        exit(1);
      }
      model_file.close();

      size_t pos = file.rfind("/");
      if (pos != std::string::npos)
      {
        std::string filename = file.substr(pos + 1);
        if (imported_modules.find(filename) != imported_modules.end())
          current_python_file = imported_modules[filename];
      }

      exprt model_code =
        with_ast(&model_json, [&]() { return get_block((*ast_json)["body"]); });

      convert_expression_to_code(model_code);

      // Accumulate model code
      models_block.copy_to_operands(model_code);
      current_python_file = main_python_file;
    }
    is_loading_models = false;
  }

  // Create a block to hold intrinsic assignments and load C intrinsics
  code_blockt intrinsic_block;
  load_c_intrisics(intrinsic_block);

  // Pre-register module-level variable symbols so class methods can reference
  // globals declared later in the file (Python LEGB rule).
  preregister_global_variables((*ast_json)["body"]);

  // Variables to hold user code and initialization code
  codet user_code;
  code_blockt init_code;

  // Handle --function option
  const std::string function = config.options.get_option("function");
  if (!function.empty())
  {
    /* If the user passes --function, we add only a call to the
     * respective function in __ESBMC_main instead of entire Python program
     */

    nlohmann::json function_node;
    // Find function node in AST
    for (const auto &element : (*ast_json)["body"])
    {
      if (element["_type"] == "FunctionDef" && element["name"] == function)
      {
        function_node = element;
        break;
      }
    }

    if (function_node.empty())
      throw std::runtime_error("Function " + function + " not found");

    code_blockt block;

    // Add intrinsic assignments first
    block.copy_to_operands(intrinsic_block);

    // Convert classes referenced by the function
    for (const auto &clazz : global_scope_.classes())
    {
      const auto &class_node = find_class((*ast_json)["body"], clazz);
      get_class_definition(class_node, block);
      current_class_name_.clear();
    }

    // Convert only the global variables referenced by the function
    for (const auto &global_var : global_scope_.variables())
    {
      const auto &var_node = find_var_decl(global_var, "", *ast_json);
      get_var_assign(var_node, block);
    }

    // Convert function arguments types
    for (const auto &arg : function_node["args"]["args"])
    {
      // Check if annotation exists and is not null before accessing "id"
      if (
        arg.contains("annotation") && !arg["annotation"].is_null() &&
        arg["annotation"].contains("id"))
      {
        auto node = find_class((*ast_json)["body"], arg["annotation"]["id"]);
        if (!node.empty())
          get_class_definition(node, block);
      }
    }

    // Convert a single function
    get_function_definition(function_node);

    // Get function symbol
    symbol_id sid = create_symbol_id();
    sid.set_function(function);
    symbolt *symbol = symbol_table_.find_symbol(sid.to_string());

    if (!symbol)
      throw std::runtime_error("Symbol " + sid.to_string() + " not found");

    // Create function call
    code_function_callt call;
    call.location() = symbol->location;
    call.function() = symbol_expr(*symbol);

    const code_typet::argumentst &arguments =
      to_code_type(symbol->get_type()).arguments();

    // Function args are nondet values
    for (const code_typet::argumentt &arg : arguments)
    {
      exprt arg_value = exprt("sideeffect", arg.type());
      arg_value.statement("nondet");
      call.arguments().push_back(arg_value);
    }

    convert_expression_to_code(call);
    convert_expression_to_code(block);

    // Prepare user code: class definitions + function call
    code_blockt user_code_body;
    user_code_body.copy_to_operands(block);
    user_code_body.copy_to_operands(call);
    user_code.swap(user_code_body);

    // Add models to init code
    if (!models_block.operands().empty())
      init_code.copy_to_operands(models_block);
  }
  else
  {
    // Convert imported modules
    module_locator locator((*ast_json)["ast_output_dir"].get<std::string>());

    // Pre-walk the import graph so each annotator can see subscript usages
    // from any other module (GitHub #4554).
    pre_collect_module_asts(*ast_json, locator);

    // Accumulate all imports
    code_blockt all_imports_block;

    for (const auto &elem : (*ast_json)["body"])
    {
      if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
      {
        if (elem.value("module_not_found", false))
        {
          const std::string module_name = (elem["_type"] == "ImportFrom")
                                            ? elem["module"]
                                            : elem["names"][0]["name"];
          log_warning("skipping unresolvable import: {}", module_name);
          continue;
        }
        is_importing_module = true;
        if (!import_module_into_block(elem, locator, all_imports_block))
        {
          const std::string &module_name = (elem["_type"] == "ImportFrom")
                                             ? elem["module"]
                                             : elem["names"][0]["name"];
          throw std::runtime_error(
            "Cannot open file: " + locator.module_path(module_name));
        }
      }
    }

    // Do the same for imports that appear directly inside functions.
    for (const auto &elem : (*ast_json)["body"])
    {
      if (
        elem["_type"] != "FunctionDef" || !elem.contains("body") ||
        !elem["body"].is_array())
        continue;

      for (const auto &stmt : elem["body"])
      {
        if (stmt["_type"] != "ImportFrom" && stmt["_type"] != "Import")
          continue;

        is_importing_module = true;
        if (!import_module_into_block(stmt, locator, all_imports_block))
        {
          const std::string &module_name = (stmt["_type"] == "ImportFrom")
                                             ? stmt["module"]
                                             : stmt["names"][0]["name"];
          throw std::runtime_error(
            "Cannot open file: " + locator.module_path(module_name));
        }
      }
    }

    is_importing_module = false;
    current_python_file = main_python_file;

    // Convert main statements
    exprt main_block = get_block((*ast_json)["body"]);
    user_code = convert_expression_to_code(main_block);

    // Prepare initialization code: models + intrinsics + imports
    if (!models_block.operands().empty())
      init_code.copy_to_operands(models_block);
    init_code.copy_to_operands(intrinsic_block);
    if (!all_imports_block.operands().empty())
      init_code.copy_to_operands(all_imports_block);
  }

  /*
   * Create three-function architecture for coverage support (similar to Solidity Frontend):
   *
   * 1. python_init
   *    - Contains models, intrinsics, and imports initialization
   *    - Marked with __ESBMC_HIDE label to exclude from coverage statistics
   *    - Only created if there is initialization code
   *
   * 2. python_user_main
   *    - Contains only user code from the main module
   *    - This is what gets analyzed for branch/decision/assertion coverage
   *
   * 3. __ESBMC_main
   *    - Entry point for ESBMC verification
   *    - Initializes static lifetime variables
   *    - Calls python_init() if it exists
   *    - Calls python_user_main()
   *
   * This architecture ensures that coverage analysis only counts user code,
   * not initialization/library code, making Python behave consistently with C.
   */
  if (!init_code.operands().empty())
  {
    code_typet init_type;
    init_type.return_type() = empty_typet();

    symbolt init_symbol;
    init_symbol.id = "python_init";
    init_symbol.name = "python_init";
    init_symbol.get_type() = init_type;
    init_symbol.lvalue = true;
    init_symbol.is_extern = false;
    init_symbol.file_local = false;
    init_symbol.location = get_location_from_decl(*ast_json);

    // Add __ESBMC_HIDE label to hide from coverage
    code_labelt esbmc_hide;
    esbmc_hide.set_label("__ESBMC_HIDE");
    esbmc_hide.code() = code_skipt();

    code_blockt init_body;
    init_body.copy_to_operands(esbmc_hide);
    init_body.copy_to_operands(init_code);
    init_symbol.get_value().swap(init_body);

    if (symbol_table_.move(init_symbol))
    {
      throw std::runtime_error("The python_init function is already defined");
    }
  }

  // Create python_user_main function containing only user code
  code_typet user_main_type;
  user_main_type.return_type() = empty_typet();

  symbolt user_main_symbol;
  user_main_symbol.id = "python_user_main";
  user_main_symbol.name = "python_user_main";
  user_main_symbol.get_type() = user_main_type;
  user_main_symbol.lvalue = true;
  user_main_symbol.is_extern = false;
  user_main_symbol.file_local = false;
  user_main_symbol.location = get_location_from_decl(*ast_json);
  user_main_symbol.get_value() = user_code;

  if (symbol_table_.move(user_main_symbol))
  {
    throw std::runtime_error(
      "The python_user_main function is already defined");
  }

  // Create __ESBMC_main that initializes and calls user code
  code_typet main_type;
  main_type.return_type() = empty_typet();

  symbolt main_symbol;
  main_symbol.id = "__ESBMC_main";
  main_symbol.name = "__ESBMC_main";
  main_symbol.get_type() = main_type;
  main_symbol.lvalue = true;
  main_symbol.is_extern = false;
  main_symbol.file_local = false;
  main_symbol.location = get_location_from_decl(*ast_json);

  code_blockt main_body;

  // 1. Initialize static lifetime variables
  symbol_table_.foreach_operand_in_order([&main_body](const symbolt &s) {
    if (s.static_lifetime && !s.get_value().is_nil() && !s.get_type().is_code())
    {
      code_assignt assign(symbol_expr(s), s.get_value());
      assign.location() = s.location;
      main_body.copy_to_operands(assign);
    }
  });

  // 2. Call python_init for initialization
  if (!init_code.operands().empty())
  {
    const symbolt *init_sym = symbol_table_.find_symbol("python_init");
    if (init_sym)
    {
      code_function_callt init_call;
      init_call.function() = symbol_expr(*init_sym);
      main_body.copy_to_operands(init_call);
    }
  }

  // 3. Bracket the user-code call with the pthread main-thread hooks
  // (parallel to the C frontend in clang_c_main.cpp). Without these
  // __ESBMC_num_threads_running stays at zero on the main thread,
  // which fires the deadlock detector inside pthread_join_switch /
  // __pyt_join as soon as any spawned thread is joined.
  //
  // The hook bodies are linked in from pthread_lib.c via the
  // python_c_models whitelist (cprover_library.cpp), but the Python
  // symbol table does not learn about them until they are referenced;
  // register the symbols here so the GOTO converter can resolve the
  // call targets. Unconditional registration is safe: the hooks have
  // empty side effects on sequential programs (they only bump
  // __ESBMC_num_threads_running), and matches the C frontend, which
  // also brackets every program regardless of whether it spawns
  // threads.
  code_typet hook_type;
  hook_type.return_type() = empty_typet();
  locationt hook_location = get_location_from_decl(*ast_json);
  auto make_hook_call = [&](const std::string &name) {
    ensure_void_void_intrinsic(name, hook_location);
    code_function_callt call;
    call.function() = symbol_exprt("c:@F@" + name, hook_type);
    call.location() = hook_location;
    return call;
  };

  // __ESBMC_yield is dereferenced unconditionally by goto_convert's
  // do_atomic_begin (builtin_functions.cpp). The __pyt_init_tid /
  // __pyt_terminate / __pyt_join bodies that the cprover_library pulls
  // in all contain __ESBMC_atomic_begin() calls, so goto_convert will
  // need the yield symbol regardless of whether user code spawns a
  // thread. Register it once here, mirroring the lazy registration
  // function_call_builder does for atomic_begin in user code.
  ensure_void_void_intrinsic("__ESBMC_yield", hook_location);

  // The threading.Lock deadlock-aware acquire (models/threading_deadlock.py,
  // loaded when ``--deadlock-check`` is set) calls this intrinsic on the
  // blocked branch. Register unconditionally: under the assume-only Lock
  // variant the symbol is unreferenced and contributes no GOTO code.
  ensure_void_void_intrinsic("__ESBMC_pylock_block_and_check", hook_location);

  main_body.copy_to_operands(make_hook_call("__ESBMC_pthread_start_main_hook"));

  // 4. Call python_user_main
  const symbolt *user_main_sym = symbol_table_.find_symbol("python_user_main");
  if (!user_main_sym)
  {
    throw std::runtime_error("python_user_main symbol not found after move");
  }

  code_function_callt user_main_call;
  user_main_call.function() = symbol_expr(*user_main_sym);
  main_body.copy_to_operands(user_main_call);

  main_body.copy_to_operands(make_hook_call("__ESBMC_pthread_end_main_hook"));

  main_symbol.get_value().swap(main_body);

  if (symbol_table_.move(main_symbol))
  {
    throw std::runtime_error(
      "The main function is already defined in another module");
  }
}
