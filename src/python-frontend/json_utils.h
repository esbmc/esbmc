#pragma once

namespace json_utils
{
template <typename JsonType>
bool is_class(const std::string &name, const JsonType &json)
{
  for (const auto &elem : json)
  {
    if (elem["_type"] == "ClassDef" && elem["name"] == name)
      return true;
  }
  return false;
}

template <typename JsonType>
JsonType find_class(const JsonType &json, const std::string &class_name)
{
  for (const auto &elem : json)
  {
    if (elem["_type"] == "ClassDef" && elem["name"] == class_name)
      return elem;
  }
  return JsonType();
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
