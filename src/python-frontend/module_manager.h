#pragma once

#include <string>
#include <memory>
#include <unordered_set>

#include <filesystem>

class module;

using ModulePtr = std::shared_ptr<module>;
using ModulesList = std::unordered_set<ModulePtr>;

class module_manager
{
public:
  static std::shared_ptr<module_manager> create(
    const std::string &module_search_path,
    const std::string &main_module_path);

  const ModulePtr get_module(const std::string &module_name) const;

private:
  module_manager(const std::string &path);
  void load();

  void load_directory(
    const std::filesystem::path &current_path,
    ModulePtr parent_module = nullptr);

  ModulePtr get_module_from_dir(
    const std::filesystem::path &path,
    ModulePtr parent_module);

  const std::string &module_search_path_;
  std::string main_module_;
  ModulesList modules_;
};
