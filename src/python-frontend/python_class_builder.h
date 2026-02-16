#pragma once

#include <nlohmann/json.hpp>
#include <python-frontend/python_class.h>

class python_converter;
struct codet;
class symbolt;
class struct_typet;

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

  void get_members(struct_typet &st, codet &out);

  void add_self_attrs(struct_typet &st);

  void gen_ctor(bool has_ud_base, struct_typet &st);

  /// Check if this class inherits from TypedDict
  bool is_typeddict_class() const;
};
