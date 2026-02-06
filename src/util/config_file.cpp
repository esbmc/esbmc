#include "config_file.h"

#include <boost/program_options/option.hpp>
#include <stdexcept>
#include <string>
#include <set>
#include <fmt/core.h>
#include <util/message.h>

#include "lib/toml.hpp"

static const std::string toml_type_to_string(const toml::node &node)
{
  using namespace toml;

  if (node.is_string())
    return "string";
  else if (node.is_integer())
    return "integer";
  else if (node.is_floating_point())
    return "floating-point";
  else if (node.is_boolean())
    return "boolean";
  else if (node.is_date())
    return "date";
  else if (node.is_time())
    return "time";
  else if (node.is_date_time())
    return "date_time";
  else if (node.is_array())
    return "array";
  else if (node.is_table())
    return "table";
  else
    return "unknown";
}

boost::program_options::basic_parsed_options<char> parse_toml_file(
  std::basic_istream<char> &is,
  const boost::program_options::options_description &desc)
{
  toml::table tbl;
  try
  {
    tbl = toml::parse(is);
  }
  catch (const toml::parse_error &err)
  {
    throw std::runtime_error(
      fmt::format("config: error parsing TOML file: {}", err.what()));
  }

  // Get all the long option names and use those as allowed options.
  // Also track which options are boolean flags (zero-token options).
  std::set<std::string> allowed_options;
  std::set<std::string> bool_flag_options;
  const std::vector<
    boost::shared_ptr<boost::program_options::option_description>> &options =
    desc.options();
  for (unsigned i = 0; i < options.size(); ++i)
  {
    const boost::program_options::option_description &d = *options[i];

    if (d.long_name().empty())
      boost::throw_exception(boost::program_options::error(
        "abbreviated option names are not permitted in options "
        "configuration files"));

    allowed_options.insert(d.long_name());
    if (d.semantic()->max_tokens() == 0)
      bool_flag_options.insert(d.long_name());
  }

  // Parser return char strings
  boost::program_options::parsed_options result(&desc);
  for (const auto &key_name : allowed_options)
  {
    if (tbl.contains(key_name))
    {
      auto value_node = tbl.get(key_name);

      auto add_option = [&](const std::string &key, const std::string &val,
                            bool quote = false) {
        if (quote)
          log_status("[CONFIG] loaded {} = \"{}\"", key, val);
        else
          log_status("[CONFIG] loaded {} = {}", key, val);
        result.options.push_back(boost::program_options::option(
          key, std::vector<std::string>(1, val)));
      };

      switch (value_node->type())
      {
      case toml::node_type::string:
      {
        if (bool_flag_options.count(key_name))
          throw std::runtime_error(fmt::format(
            "config: key '{}' is a boolean flag but got string value \"{}\""
            " (use {} = true instead of {} = \"true\")",
            key_name,
            value_node->as_string()->get(),
            key_name,
            key_name));

        add_option(key_name, value_node->as_string()->get(), true);
        break;
      }
      case toml::node_type::integer:
      {
        add_option(key_name, std::to_string(value_node->as_integer()->get()));
        break;
      }
      case toml::node_type::floating_point:
      {
        add_option(
          key_name,
          std::to_string(value_node->as_floating_point()->get()));
        break;
      }
      case toml::node_type::boolean:
      {
        const auto value = value_node->as_boolean()->get();
        log_status(
          "[CONFIG] loaded {} = {}", key_name, value ? "true" : "false");
        // Boolean flags are handled in this codebase as flags, so only add if
        // true. For context: cmdlinet::isset
        // Also they are added as blank strings!
        if (value)
        {
          result.options.push_back(boost::program_options::option(
            key_name, std::vector<std::string>(1, "")));
        }
        break;
      }
      // Not supported types currently.
      case toml::node_type::table:
      case toml::node_type::array:
      case toml::node_type::date:
      case toml::node_type::time:
      case toml::node_type::date_time:
      case toml::node_type::none:
        throw std::runtime_error(fmt::format(
          "config: invalid key type: {}: {}",
          key_name,
          toml_type_to_string(*value_node)));
      };
    }
  }

  // Convert char strings into desired type.
  return boost::program_options::basic_parsed_options<char>(result);
}
