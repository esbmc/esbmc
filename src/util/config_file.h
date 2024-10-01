#pragma once

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/detail/config_file.hpp>

/* Contains code for parsing and loading files. */

boost::program_options::basic_parsed_options<char> parse_toml_file(
  std::basic_istream<char> &is,
  const boost::program_options::options_description &desc,
  bool allow_unregistered = false);
