#pragma once

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/detail/config_file.hpp>

template <class charT = char>
boost::program_options::basic_parsed_options<charT> parse_toml_file(
  std::basic_istream<charT> &,
  const boost::program_options::options_description &,
  bool allow_unregistered = false);