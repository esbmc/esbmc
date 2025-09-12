#pragma once

#include <string>

class symbol_id
{
public:
  symbol_id(
    const std::string &file,
    const std::string &clazz,
    const std::string &function);

  symbol_id() = default;

  static symbol_id from_string(const std::string &str);

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

  void set_filename(const std::string &filename)
  {
    filename_ = filename;
  }

  const std::string &get_prefix() const
  {
    return prefix_;
  }

  const std::string &get_function() const
  {
    return function_name_;
  }

  const std::string &get_class() const
  {
    return classname_;
  }

  const std::string &get_filename() const
  {
    return filename_;
  }

  const std::string &get_object() const
  {
    return object_;
  }

  void clear();

  std::string to_string() const;

  std::string global_to_string() const;

private:
  std::string filename_;
  std::string classname_ = "";
  std::string function_name_ = "";
  std::string object_ = "";
  std::string attribute_ = "";
  std::string prefix_ = "py:";
};
