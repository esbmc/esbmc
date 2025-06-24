#include <python-frontend/module_manager.h>
#include <python-frontend/module.h>
#include <python-frontend/json_utils.h>
#include <util/message.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

module_manager::module_manager(const std::string &module_search_path)
  : module_search_path_(module_search_path)
{
}

std::shared_ptr<module> create_module(const fs::path &json_path)
{
  std::ifstream json_file(json_path);
  if (!json_file.is_open())
    return nullptr;

  auto md = std::make_shared<module>(json_path.stem().string());

  try
  {
    nlohmann::json ast;
    json_file >> ast;
    json_file.close();

    for (const auto &node : ast["body"])
    {
      if (node["_type"] == "FunctionDef")
      {
        function f;
        f.name_ = node["name"];

        if (node["returns"].is_null())
          continue;

        if (node["returns"]["_type"] == "Subscript")
          f.return_type_ = node["returns"]["value"]["id"];
        else if (node["returns"]["_type"] == "Tuple")
          f.return_type_ = "Tuple";
<<<<<<< HEAD
        else if (
          node["returns"]["_type"] == "Constant" ||
          node["returns"]["_type"] == "Str")
          // Handle string annotations like -> "int" (legacy forward references)
          f.return_type_ = node["returns"]["value"];
=======
>>>>>>> 702b6f4ad ([python] improved return type inference (#2494))
        else if (
          node["returns"]["_type"] == "Constant" ||
          node["returns"]["_type"] == "Str")
          // Handle string annotations like -> "int" (legacy forward references)
          f.return_type_ = node["returns"]["value"];
        else if (
          node["returns"].contains("value") &&
          node["returns"]["value"].is_null())
          f.return_type_ = "None";
        else
          f.return_type_ = node["returns"]["id"];

        md->add_function(f);
      }
    }

    return md;
  }
  catch (const nlohmann::json::parse_error &e)
  {
    // Catches JSON parsing errors (e.g., invalid JSON content)
    log_error("Error parsing the JSON: {}", e.what());
    return nullptr;
  }
  catch (const std::exception &e)
  {
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
          continue;

        auto current_module = get_module(submodule->name());
        if (current_module)
        {
          current_module->add_functions(submodule->functions());
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
  return get_module_recursive(parts, modules_);
}
