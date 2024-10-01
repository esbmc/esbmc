#include "config_file.h"

#include <boost/program_options/option.hpp>
#include <stdexcept>
#include <string>
#include <set>
#include <format>

#include "fmt/color.h"
#include "lib/toml.hpp"
#include "util/message.h"

// Define this because we cant use log_debug due to this being too early in init.
#define DEBUG_MSG 0
#if DEBUG_MSG
#  define LOG_MSG(fmt, values...) log_status(fmt, values)
#else
#  define LOG_MSG(fmt, values...)
#endif

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
  const boost::program_options::options_description &desc,
  bool allow_unregistered)
{
  toml::table tbl;
  try
  {
    tbl = toml::parse(is);
  }
  catch (const toml::parse_error &err)
  {
    LOG_MSG("Config: error parsing TOML file: %s", err.what());
    throw err;
  }

  // Get all the long option names and use those as allowed options.
  std::set<std::string> allowed_options;
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
  }

  // Parser return char strings
  boost::program_options::parsed_options result(&desc);
  for (const auto &key_name : allowed_options)
  {
    if (tbl.contains(key_name))
    {
      auto value_node = tbl.get(key_name);
      switch (value_node->type())
      {
      case toml::node_type::string:
      case toml::node_type::integer:
      case toml::node_type::floating_point:
      {
        const auto value = std::to_string(value_node->as_integer()->get());
        // For some reason takes it in as an array of values.
        const auto option = boost::program_options::option(
          key_name, std::vector<std::string>(1, value));
        result.options.push_back(option);
        LOG_MSG("Config: Parsed option: {} {}", key_name, value);
        break;
      }
      case toml::node_type::boolean:
      {
        const auto value = value_node->as_boolean()->get();
        // Boolean flags are handled in this codebase as flags, so only add if
        // true. For context: cmdlinet::isset
        if (value)
        {
          const auto option = boost::program_options::option(
            key_name, std::vector<std::string>(1, ""));
        }
        LOG_MSG("Config: Parsed boolean option: {} {}", key_name, value);
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
        break;
      };
    }
  }

  if (!allow_unregistered)
  {
    // Check for additional options.
    for (auto node = tbl.cbegin(); node != tbl.cend(); ++node)
    {
      const auto key = std::string(node->first.str());
      const bool is_in = allowed_options.find(key) != allowed_options.end();
      if (!is_in)
      {
        throw std::runtime_error(fmt::format("Config: Invalid key: {}", key));
      }
    }
  }

  // Convert char strings into desired type.
  return boost::program_options::basic_parsed_options<char>(result);
}
