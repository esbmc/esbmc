#include <python-frontend/module_manager.h>
#include <python-frontend/module.h>
#include <python-frontend/json_utils.h>
#include <util/message.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace
{
// Helper function to safely extract string value from JSON node
// Returns empty string if the field is missing, null, or not a string
std::string get_string_safe(const nlohmann::json &node, const std::string &key)
{
  if (!node.contains(key))
    return "";

  if (node[key].is_string())
    return node[key].get<std::string>();

  return "";
}
} // namespace

module_manager::module_manager(const std::string &module_search_path)
  : module_search_path_(module_search_path)
{
}

std::shared_ptr<module> create_module(const fs::path &json_path)
{
  std::ifstream json_file(json_path);
  if (!json_file.is_open())
  {
    log_warning(
      "[module_manager] create_module: failed to open {}", json_path.string());
    return nullptr;
  }

  std::string module_name = json_path.stem().string();
  auto md = std::make_shared<module>(module_name);

  try
  {
    nlohmann::json ast;
    json_file >> ast;
    json_file.close();

    // Validate JSON structure
    if (!ast.contains("body") || !ast["body"].is_array())
    {
      log_error(
        "[module_manager] create_module: Invalid or missing 'body' in {}",
        json_path.string());
      return nullptr;
    }

    for (const auto &node : ast["body"])
    {
      std::string node_type =
        node.contains("_type") && node["_type"].is_string()
          ? node["_type"].get<std::string>()
          : "unknown";

      if (node_type == "unknown")
      {
        log_warning(
          "[module_manager] create_module: Unknown or missing node type in {}",
          json_path.string());
        continue;
      }

      if (node_type == "FunctionDef")
      {
        function f;
        f.name_ = get_string_safe(node, "name");
        if (f.name_.empty())
        {
          continue;
        }

        if (node["returns"].is_null())
          continue;

        // Handle PEP 604 union syntax: int | bool
        if (node["returns"]["_type"] == "BinOp")
          f.return_type_ = "Union";
        else if (node["returns"]["_type"] == "Subscript")
        {
          if (
            node["returns"].contains("value") &&
            node["returns"]["value"].contains("id") &&
            node["returns"]["value"]["id"].is_string())
            f.return_type_ = node["returns"]["value"]["id"].get<std::string>();
        }
        else if (node["returns"]["_type"] == "Tuple")
          f.return_type_ = "Tuple";
        else if (
          node["returns"]["_type"] == "Constant" ||
          node["returns"]["_type"] == "Str")
        {
          // Handle string annotations like -> "int" (legacy forward references)
          if (node["returns"].contains("value"))
          {
            if (node["returns"]["value"].is_string())
              f.return_type_ = node["returns"]["value"].get<std::string>();
            else if (node["returns"]["value"].is_null())
              f.return_type_ = "None";
          }
        }
        else if (
          node["returns"].contains("value") &&
          node["returns"]["value"].is_null())
          f.return_type_ = "None";
        else if (
          node["returns"].contains("id") && node["returns"]["id"].is_string())
          f.return_type_ = node["returns"]["id"].get<std::string>();
        else
          f.return_type_ = "None";

        if (json_utils::has_overload_decorator(node))
          md->add_overload(node);

        md->add_function(f);
      }
      else if (node_type == "ClassDef")
      {
        class_definition c;

        // Safely get class name
        c.name_ = get_string_safe(node, "name");
        if (c.name_.empty())
        {
          continue;
        }

        // Process base classes
        if (node.contains("bases") && node["bases"].is_array())
        {
          for (const auto &base : node["bases"])
          {
            std::string base_name = get_string_safe(base, "id");
            if (!base_name.empty())
              c.bases_.push_back(base_name);
          }
        }

        // Process methods
        if (node.contains("body") && node["body"].is_array())
        {
          for (const auto &item : node["body"])
          {
            if (item["_type"] == "FunctionDef")
            {
              std::string method_name = get_string_safe(item, "name");
              if (!method_name.empty())
                c.methods_.push_back(method_name);
            }
          }
        }

        md->add_class(c);
      }
    }

    return md;
  }
  catch (const nlohmann::json::type_error &e)
  {
    log_error(
      "JSON type error in create_module for {}: {} (id: {})",
      json_path.string(),
      e.what(),
      e.id);
    return nullptr;
  }
  catch (const nlohmann::json::parse_error &e)
  {
    // Catches JSON parsing errors (e.g., invalid JSON content)
    log_error("Error parsing the JSON {}: {}", json_path.string(), e.what());
    return nullptr;
  }
  catch (const std::exception &e)
  {
    log_error(
      "Exception in create_module for {}: {}", json_path.string(), e.what());
    return nullptr;
  }
}

void module_manager::load_directory(
  const fs::path &current_path,
  ModulePtr parent_module)
{
  for (const auto &entry : fs::directory_iterator(current_path))
  {
    if (entry.is_regular_file() && entry.path().extension() == ".json")
    {
      // Create a module for the JSON file
      auto submodule = create_module(entry.path());
      if (submodule)
      {
        if (main_module_ == submodule->name())
        {
          continue;
        }

        auto current_module = get_module(submodule->name());
        if (current_module)
        {
          current_module->add_functions(submodule->functions());
          current_module->add_classes(submodule->classes());
          continue;
        }

        if (parent_module)
        {
          parent_module->add_submodule(submodule); // Add to the parent module
        }
        else
        {
          modules_.insert(submodule); // Add to the top level
        }
      }
    }
    else if (entry.is_directory())
    {
      // Create or retrieve the module corresponding to the directory
      auto module_dir = get_module_from_dir(entry.path(), parent_module);

      // Recursively process files and subdirectories
      if (module_dir)
        load_directory(entry.path(), module_dir);
    }
  }
}

ModulePtr module_manager::get_module_from_dir(
  const fs::path &path,
  ModulePtr parent_module)
{
  ModulePtr current_module = parent_module;

  auto relative_path = std::filesystem::relative(path, module_search_path_);

  if (relative_path.filename() == "__pycache__")
    return nullptr;

  // Split the path into components and process each one
  for (const auto &component : relative_path)
  {
    // Module name without extension
    const auto module_name = component.stem().string();
    if (!current_module)
    {
      // If there is no parent module, create or get it at the top level
      current_module = get_module(module_name);
      if (!current_module && !module_name.empty())
      {
        current_module = std::make_shared<module>(module_name);
        modules_.insert(current_module);
      }
    }
    else
    {
      // If there is a parent module, create or get it as a submodule
      auto existing_submodule = std::find_if(
        current_module->submodules().begin(),
        current_module->submodules().end(),
        [&module_name](std::shared_ptr<module> mod) {
          return mod->name() == module_name;
        });

      if (existing_submodule == current_module->submodules().end())
      {
        auto new_submodule = std::make_shared<module>(module_name);
        current_module->add_submodule(new_submodule);
        current_module = new_submodule;
      }
      else
      {
        current_module = *existing_submodule;
      }
    }
  }

  return current_module;
}

void module_manager::load()
{
  load_directory(module_search_path_);
}

std::shared_ptr<module_manager> module_manager::create(
  const std::string &module_search_path,
  const std::string &main_module_path)
{
  std::shared_ptr<module_manager> mm(new module_manager(module_search_path));
  if (mm)
  {
    mm->main_module_ = fs::path(main_module_path).stem().string();
    mm->load();
    return mm;
  }
  return nullptr;
}

static std::vector<std::string> split(const std::string &str, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter))
    tokens.push_back(token);

  return tokens;
}

ModulePtr get_module_recursive(
  const std::vector<std::string> &parts,
  const ModulesList &current_modules)
{
  if (parts.empty())
    return nullptr;

  // First, find the module that matches the first part of the module name
  for (const auto &mod : current_modules)
  {
    if (mod->name() == parts[0])
    {
      if (parts.size() == 1)
      {
        // If there's no more parts left, return the found module
        return mod;
      }
      else
      {
        // If there are more parts, search in the submodules
        return get_module_recursive(
          std::vector<std::string>(parts.begin() + 1, parts.end()),
          mod->submodules());
      }
    }
  }
  return nullptr; // Return nullptr if the module is not found
}

const ModulePtr module_manager::get_module(const std::string &module_name) const
{
  std::vector<std::string> parts = split(module_name, '.');
  auto result = get_module_recursive(parts, modules_);
  return result;
}
