#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/// Dedicated cache for function-call resolution data.
/// Holds inferred class-type candidates and method-existence results,
/// isolating cache responsibilities from `function_call_expr` internals.
class function_call_cache
{
public:
  // ---- possible class types cache ----

  std::optional<std::vector<std::string>>
  get_possible_class_types(const std::string &key) const
  {
    auto it = possible_class_types_cache_.find(key);
    if (it != possible_class_types_cache_.end())
      return it->second;
    return std::nullopt;
  }

  void set_possible_class_types(
    const std::string &key,
    const std::vector<std::string> &types)
  {
    possible_class_types_cache_[key] = types;
  }

  // ---- method exists cache ----

  std::optional<bool> get_method_exists(const std::string &key) const
  {
    auto it = method_exists_cache_.find(key);
    if (it != method_exists_cache_.end())
      return it->second;
    return std::nullopt;
  }

  void set_method_exists(const std::string &key, bool exists)
  {
    method_exists_cache_[key] = exists;
  }

  // ---- lifecycle ----

  void clear()
  {
    possible_class_types_cache_.clear();
    method_exists_cache_.clear();
  }

private:
  std::unordered_map<std::string, std::vector<std::string>>
    possible_class_types_cache_;
  std::unordered_map<std::string, bool> method_exists_cache_;
};
