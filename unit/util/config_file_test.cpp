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
    desc->add_options()("floatbv,b", po::value<bool>()->default_value(false), "")(
      "unlimited-k-steps,s", po::value<std::string>(), "")(
      "context-bound,c", po::value<int>()->default_value(2), "")(
      "verbosity,v", po::value<int>()->default_value(0), "Verbosity level");
  }

  ~TestFixture()
  {
    delete desc;
  }
};

std::string const right_conf_file =
  "floatbv = true\n"
  "unlimited-k-steps = false\n"
  "memory-leak-check = true\n"
  "context-bound = 2\n"
  "verbosity = '999'";

std::string const wrong_conf_file =
  "floatbv = true\n"
  "unlimited-k-steps = false\n"
  "memory-leak-check = true\n"
  "context-bound = 2\n"
  "fake = 'hello'";

std::string const wrong_conf_file_2 = "b = true\n";

void check_option(boost::program_options::option option, std::string key, std::string value) {
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
      parse_toml_file(iss, *desc, false);
    REQUIRE(options.options.size() == 0);
  }

  SECTION("Load values allow unregistered")
  {
    std::istringstream iss(right_conf_file);
    boost::program_options::basic_parsed_options<char> options =
      parse_toml_file(iss, *desc, true);

    REQUIRE(options.options.size() == 5);
    check_option(options.options[0], "floatbv", "true");
    check_option(options.options[1], "unlimited-k-steps", "false");
    check_option(options.options[2], "memory-leak-check", "true");
    check_option(options.options[3], "context-bound", "2");
    check_option(options.options[4], "verbosity", "999");
  }

  SECTION("Load values prohibit unregistered")
  {
    std::istringstream iss(wrong_conf_file);
    REQUIRE_THROWS_AS(parse_toml_file(iss, *desc, false), std::runtime_error);
  }

  SECTION("Fail with short option")
  {
    std::istringstream iss(wrong_conf_file_2);
    REQUIRE_THROWS_AS(parse_toml_file(iss, *desc, false), std::runtime_error);
  }
}
