#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <util/config.h>
#include <util/std_code.h>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

codet python_converter::convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return static_cast<codet &>(expr);

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);
  return code;
}

symbolt python_converter::create_symbol(
  const std::string &module,
  const std::string &name,
  const std::string &id,
  const locationt &location,
  const typet &type) const
{
  symbolt symbol;
  symbol.mode = "Python";
  symbol.module = module;
  symbol.location = location;
  symbol.set_type(type);
  symbol.name = name;
  symbol.id = id;
  return symbol;
}

void python_converter::ensure_void_void_intrinsic(
  const std::string &name,
  const locationt &location)
{
  const std::string symbol_id = "c:@F@" + name;
  if (symbol_table_.find_symbol(symbol_id) != nullptr)
    return;
  code_typet fn_type;
  fn_type.return_type() = empty_typet();
  symbolt symbol =
    create_symbol(python_file(), name, symbol_id, location, fn_type);
  add_symbol(symbol);
}

symbol_id python_converter::create_symbol_id(const std::string &filename) const
{
  return symbol_id(filename, current_class_name_, current_func_name_);
}

symbol_id python_converter::create_symbol_id() const
{
  return symbol_id(
    current_python_file, current_class_name_, current_func_name_);
}

locationt
python_converter::get_location_from_decl(const nlohmann::json &ast_node) const
{
  locationt location;
  if (ast_node.contains("lineno"))
    location.set_line(ast_node["lineno"].get<int>());

  if (ast_node.contains("col_offset"))
    location.set_column(ast_node["col_offset"].get<int>());

  location.set_file(current_python_file.c_str());
  location.set_function(current_func_name_);
  return location;
}

void python_converter::copy_location_fields_from_decl(
  const nlohmann::json &from,
  nlohmann::json &to) const
{
  const locationt loc = get_location_from_decl(from);
  const std::string line = id2string(loc.get_line());
  if (!line.empty())
    to["lineno"] = std::stoi(line);

  const std::string column = id2string(loc.get_column());
  if (!column.empty())
    to["col_offset"] = std::stoi(column);

  if (from.contains("end_lineno"))
    to["end_lineno"] = from["end_lineno"];
  if (from.contains("end_col_offset"))
    to["end_col_offset"] = from["end_col_offset"];
}

bool python_converter::type_assertions_enabled() const
{
  return config.options.get_bool_option("is-instance-check");
}

bool python_converter::is_coverage_mode() const
{
  return config.options.get_bool_option("condition-coverage") ||
         config.options.get_bool_option("condition-coverage-claims") ||
         config.options.get_bool_option("condition-coverage-rm") ||
         config.options.get_bool_option("condition-coverage-claims-rm") ||
         config.options.get_bool_option("branch-coverage") ||
         config.options.get_bool_option("branch-coverage-claims") ||
         config.options.get_bool_option("branch-function-coverage") ||
         config.options.get_bool_option("branch-function-coverage-claims");
}

bool python_converter::is_pytest_generation_mode() const
{
  return config.options.get_bool_option("generate-pytest-testcase");
}

bool python_converter::is_model_file(const nlohmann::json &node) const
{
  const std::string file = get_location_from_decl(node).file().as_string();
  // The file under verification is never a model — not even when it is passed
  // by a bare relative name (e.g. `main.py`) that the no-directory heuristic
  // below would otherwise misread as a model. That misclassification disabled
  // `and`/`or` short-circuit lowering for the main module, so a guard like
  // `x is None or x.field is None` was emitted as an eager compound operand and
  // the verdict flipped depending on whether the source was given by relative
  // or absolute path (QuixBugs detect_cycle).
  if (file == main_python_file)
    return false;

  if (file.find("/models/") != std::string::npos)
    return true;

  if (file.find('/') == std::string::npos)
  {
    if (file.size() >= 3 && file.compare(file.size() - 3, 3, ".py") == 0)
      return true;
    return false;
  }

  const std::string suffix = "/models/" + file;
  for (const auto &entry : imported_modules)
  {
    const std::string &path = entry.second;
    if (
      path.size() >= suffix.size() &&
      path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0)
      return true;
  }

  return false;
}

void python_converter::append_models_from_directory(
  std::list<std::string> &file_list,
  const std::string &dir_path)
{
  fs::path directory(dir_path);

  // Checks if the directory exists
  if (!fs::exists(directory) || !fs::is_directory(directory))
    return;

  // Iterates over the files in the directory
  for (fs::directory_iterator it(directory), end_it; it != end_it; ++it)
  {
    if (fs::is_regular_file(*it) && it->path().extension() == ".json")
    {
      std::string file_name =
        directory.filename().string() + "/" +
        it->path().stem().string(); // File name without the extension
      file_list.push_back(file_name);

      imported_modules[it->path().stem().string()] = it->path().string();
    }
  }
}
