#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <string>
#include <vector>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/config_file.h>

struct TestFixture
{
  boost::program_options::options_description *desc;

  TestFixture()
  {
    namespace po = boost::program_options;
    desc = new boost::program_options::options_description("Allowed options");
    desc->add_options()(
      "floatbv,b", po::value<bool>()->default_value(false), "Use floatbv")(
      "unlimited-k-steps,s", po::value<std::string>(), "Use unlimited k-steps")(
      "context-bound,c",
      po::value<int>()->default_value(2),
      "Bound by x context")(
      "name,v", po::value<std::string>()->default_value("test"), "name")(
      "flag-option,f", po::bool_switch(), "A boolean flag")(
      "threshold,t", po::value<double>()->default_value(0.5), "A float value");
  }

  ~TestFixture()
  {
    delete desc;
  }
};

std::string const right_conf_file =
  "floatbv = true\n"
  "unlimited-k-steps = false\n"
  "context-bound = 2\n"
  "name = '999'";

void check_option(
  boost::program_options::option option,
  std::string key,
  std::string value)
{
  INFO("key: " << key << " value: " << value);
  REQUIRE(option.string_key == key);
  REQUIRE(option.value[0] == value);
}

TEST_CASE_METHOD(
  TestFixture,
  "config_file_general",
  "[core][util][config_file]")
{
  SECTION("Empty config file")
  {
    std::istringstream iss("");
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc);
    REQUIRE(options.options.size() == 0);
  }

  SECTION("Load values")
  {
    std::istringstream iss(right_conf_file);
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc);

    // Keys need to be in alphabetical order!
    REQUIRE(options.options.size() == 3);
    check_option(options.options[0], "context-bound", "2");
    check_option(options.options[1], "floatbv", "");
    // unlimited-k-steps will not be added because it is false
    check_option(options.options[2], "name", "999");
  }

  SECTION("String value for bool flag throws")
  {
    std::istringstream iss("flag-option = \"true\"");
    REQUIRE_THROWS_WITH(
      parse_toml_file(iss, *desc),
      Catch::Matchers::Contains(
        "use flag-option = true instead of flag-option = \"true\""));
  }

  SECTION("Bool flag with true value is loaded")
  {
    std::istringstream iss("flag-option = true");
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc);
    REQUIRE(options.options.size() == 1);
    check_option(options.options[0], "flag-option", "");
  }

  SECTION("Bool flag with false value is omitted")
  {
    std::istringstream iss("flag-option = false");
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc);
    REQUIRE(options.options.size() == 0);
  }

  SECTION("Integer value for bool flag is passed through")
  {
    std::istringstream iss("flag-option = 1");
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc);
    REQUIRE(options.options.size() == 1);
    check_option(options.options[0], "flag-option", "1");
  }

  SECTION("Float value for bool flag is passed through")
  {
    std::istringstream iss("flag-option = 1.0");
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc);
    REQUIRE(options.options.size() == 1);
    check_option(options.options[0], "flag-option", "1.000000");
  }

  SECTION("Load float values")
  {
    std::istringstream iss("threshold = 3.14");
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc);
    REQUIRE(options.options.size() == 1);
    check_option(options.options[0], "threshold", "3.140000");
  }
}
