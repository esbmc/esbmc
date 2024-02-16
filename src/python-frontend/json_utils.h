#pragma once

namespace json_utils
{

template <typename JsonType>
JsonType find_class(const JsonType &ast_json, const std::string &class_name)
{
  auto it = std::find_if(
    ast_json.begin(),
    ast_json.end(),
    [&](const JsonType &obj)
    {
      return obj.contains("_type") && obj["_type"] == "ClassDef" &&
             obj["name"] == class_name;
    });

  return (it != ast_json.end()) ? *it : JsonType();
}

template <typename JsonType>
bool is_class(const std::string &name, const JsonType &ast_json)
{
  // Find class definition in the current json
  if (find_class(ast_json, name) != JsonType())
    return true;

  auto esbmc_data = std::find_if(
    ast_json.begin(),
    ast_json.end(),
    [&](const JsonType &obj)
    { return obj["_type"] == "ESBMC" && obj.contains("ast_output_dir"); });

  if (esbmc_data == ast_json.end())
    abort();

  // Find class definition in imported modules
  for (const auto &obj : ast_json)
  {
    // Check if the current object has the _type field and its value is "ImportFrom"
    if (obj.contains("_type") && obj["_type"] == "ImportFrom")
    {
      std::stringstream module_path;
      module_path
        << esbmc_data->at("ast_output_dir").template get<std::string>() << "/"
        << obj["module"].template get<std::string>() << ".json";

      std::ifstream imported_file(module_path.str());
      JsonType imported_module_json;
      imported_file >> imported_module_json;

      if (is_class(name, imported_module_json["body"]))
        return true;
    }
  }

  return false;
}

template <typename JsonType>
JsonType find_function(const JsonType &json, const std::string &func_name)
{
  for (const auto &elem : json)
  {
    if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      return elem;
  }
  return JsonType();
}

} // namespace json_utils
