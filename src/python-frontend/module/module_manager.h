#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>

class module;

using ModulePtr = std::shared_ptr<module>;
using ModulesList = std::unordered_set<ModulePtr>;

class module_manager
{
public:
  ~module_manager();

  static std::shared_ptr<module_manager> create(
    const std::string &module_search_path,
    const std::string &main_module_path);

  /// Look a module up by (possibly dotted) name, parsing its JSON AST if this
  /// is the first request for it.
  const ModulePtr get_module(const std::string &module_name);

private:
  module_manager(const std::string &path);
  void load();

  /// get_module without the parse, for the directory walk's own lookups.
  const ModulePtr find_module(const std::string &module_name) const;

  /// Parse \p mod's sources if a lookup has not already done so. Returns
  /// false when it has sources and none of them could be read, which is what
  /// makes get_module answer nullptr rather than an empty module.
  bool hydrate(const ModulePtr &mod);

  void load_directory(
    const std::filesystem::path &current_path,
    ModulePtr parent_module = nullptr);

  ModulePtr get_module_from_dir(
    const std::filesystem::path &path,
    ModulePtr parent_module);

  std::string module_search_path_;
  std::string main_module_;
  ModulesList modules_;

  /// Module nodes the directory walk built, and how many of those carried
  /// sources a lookup has since parsed. Reported at destruction;
  /// regression/python/module_lazy_parse_budget pins the parsed count.
  unsigned discovered_ = 0;
  unsigned parsed_ = 0;
};
