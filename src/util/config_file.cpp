#include "config_file.h"

#include <string>
#include <set>

template <class charT>
boost::program_options::basic_parsed_options<charT> parse_toml_file(
  std::basic_istream<charT> &is,
  const boost::program_options::options_description &desc,
  bool allow_unregistered)
{
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
  copy(
    // TODO Replace this with TOML
    boost::program_options::detail::basic_config_file_iterator<charT>(
      is, allowed_options, allow_unregistered),
    boost::program_options::detail::basic_config_file_iterator<charT>(),
    back_inserter(result.options));
  // Convert char strings into desired type.
  return boost::program_options::basic_parsed_options<charT>(result);
}
