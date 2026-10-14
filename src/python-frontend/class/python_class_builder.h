#pragma once

#include <nlohmann/json.hpp>
#include <python-frontend/class/python_class.h>

class python_converter;
class codet;
class symbolt;
class struct_typet;

class symbol_id;

class python_class_builder
{
public:
  python_class_builder(python_converter &conv, const nlohmann::json &cls_node)
    : conv_(conv), cls_(cls_node)
  {
    pc_.parse(cls_);
  }

  void build(codet &out);

private:
  python_converter &conv_;
  const nlohmann::json &cls_;
  python_class pc_;

  // helpers
  static std::string leaf(const std::string &dotted);

  symbolt *ensure_sym(const std::string &name);

  bool get_bases(struct_typet &st);

  /// Add the converted method \p sid to \p st's method table, reporting
  /// whether it could be. A method that was not converted has no symbol, and
  /// dereferencing that miss used to segfault.
  bool register_method(struct_typet &st, const symbol_id &sid);

  void get_members(struct_typet &st, codet &out, bool has_ud_base);

  void add_self_attrs(struct_typet &st);

  void gen_ctor(bool has_ud_base, struct_typet &st);

  /// Check if this class inherits from TypedDict
  bool is_typeddict_class() const;
};
