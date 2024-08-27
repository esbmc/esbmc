#pragma once

#include <string>

class symbol_id
{
public:
  symbol_id(
    const std::string &file,
    const std::string &clazz,
    const std::string &function);

  void set_object(const std::string &obj)
  {
    object_ = obj;
  }

  void set_attribute(const std::string &attr)
  {
    attribute_ = attr;
  }

  void set_class(const std::string &clazz)
  {
    classname_ = clazz;
  }

  void set_function(const std::string &func)
  {
    function_name_ = func;
  }

  void set_prefix(const std::string &prefix)
  {
    prefix_ = prefix;
  }

  const std::string &get_function() const
  {
    return function_name_;
  }

  void clear() {
    filename_.clear();
    classname_.clear();
    function_name_.clear();
    object_.clear();
    attribute_.clear();
  }

  std::string to_string() const;

private:
  symbol_id() = delete;

  std::string filename_;
  std::string classname_ = "";
  std::string function_name_ = "";
  std::string object_ = "";
  std::string attribute_ = "";
  std::string prefix_ = "py:";
};
