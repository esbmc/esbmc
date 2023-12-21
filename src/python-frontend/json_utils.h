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

} // namespace json_utils
