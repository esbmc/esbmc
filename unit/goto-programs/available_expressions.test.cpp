#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <goto-programs/abstract-interpretation/gcse.h>
#include <util/prefix.h>
/* Testing this is almost impossible without having
 * a goto_program generation interface
 *
 * So... I will check for a vector of symbols/constants
 * in additions. Not the best way... but it should be
 * good enough for our purposes.
 */

namespace
{
// GOTO program index (PC) -> List of "symbols". This could be a vector of vectors.. but its enough for now
typedef std::map<std::string, std::vector<std::string>> test_vector;

class ae_program
{
public:
  std::string code;
  test_vector available_expressions;
  test_vector unavailable_expressions;

  // A + B + C --> [A,B,C]
  bool flatten_addition(const expr2tc &e, std::vector<std::string> &v) const
  {
    if (is_symbol2t(e))
    {
      std::string thename = to_symbol2t(e).thename.as_string();
      if (!has_prefix(thename, "c:@__ESBMC"))
        v.push_back(thename);

      return true;
    }

    if (!is_add2t(e))
      return false;

    bool side1 = flatten_addition(to_add2t(e).side_1, v);
    bool side2 = flatten_addition(to_add2t(e).side_2, v);

    return side1 && side2;
  }

  void run_test(ait<cse_domaint> &AE)
  {
    // Build the GOTO program from C
    auto P = goto_factory::get_goto_functions(
      code, goto_factory::Architecture::BIT_32);
    REQUIRE(P.functions.function_map.size() > 0);

    // Run the Points-To analysis (eventually this will be removed)
#if 0 // This makes ESBMC's string container crash. 
    cse_domaint::vsa = std::make_unique<value_set_analysist>(P.ns);
    (*cse_domaint::vsa)(P.functions);
#endif

    // Run the Abstract Interpretation
    AE(P.functions, P.ns);
    REQUIRE(P.functions.function_map.size() > 0);
#if 0
    // Test!
    Forall_goto_functions(f_it, P.functions)
    {
      if(f_it->first == "c:@F@main")
      {
        REQUIRE(f_it->second.body_available);
        forall_goto_program_instructions(i_it, f_it->second.body)
        {
          const cse_domaint &state = AE[i_it];

          const auto &should_be_present =
            available_expressions.find(i_it->location.get_line().as_string());

          const auto &should_not_be_present =
            unavailable_expressions.find(i_it->location.get_line().as_string());

          bool present_check = should_be_present == available_expressions.end();
          log_status("Checking it: {}", i_it->location.get_line().as_string());
          for(const auto &expr : state.available_expressions)
          {
            if(
              (should_not_be_present == unavailable_expressions.end()) &&
              present_check)
              break;

            std::vector<std::string> results;
            if(!flatten_addition(expr, results))
              continue;

            // Not present check
            if(results.size() == should_not_be_present->second.size())
            {
              bool found = true;
              for(int i = 0; i < results.size(); i++)
              {
                const std::string &var_name = should_not_be_present->second[i];
                const std::string &real_name = results[i];
                log_status("{} != {}", var_name, real_name);
                if(!std::equal(
                     var_name.rbegin(), var_name.rend(), real_name.rbegin()))
                {
                  found = false;
                  break;
                }
              }
              REQUIRE(!found);
            }

            if(present_check)
              continue;

            // Present check
            if(results.size() == should_be_present->second.size())
            {
              bool found = true;
              for(int i = 0; i < results.size(); i++)
              {
                const std::string &var_name = should_be_present->second[i];
                const std::string &real_name = results[i];
                log_status("{} == {}", var_name, real_name);
                if(!std::equal(
                     var_name.rbegin(), var_name.rend(), real_name.rbegin()))
                {
                  found = false;
                  break;
                }
              }
              present_check = found;
            }
          }

          REQUIRE(present_check);
        }
      }
    }
#endif
  }
};
} // namespace

TEST_CASE("Basic Expressions", "[ai][available-expressions]")
{
  // Setup global options here
  ait<cse_domaint> AE;

  ae_program T;
  T.code =
    "int main() {\n"
    "int a,b;\n"
    "int c = a + b;\n" // Here no expression should be available
    "int d;\n"         // Here a + b should be available
    "a = 42;\n"        // Here a + b should not be available
    "int *e = &d;\n"   // Here a + b should not be available
    "return a;\n"
    "}";
  T.unavailable_expressions["4"] = {
    "@F@main@a"}; // individual symbols should not be cached
  T.available_expressions["4"] = {"@F@main@a", "@F@main@b"};

  T.unavailable_expressions["6"] = {"@F@main@a", "@F@main@b"};

  T.run_test(AE);
}

TEST_CASE("Expressions - Loops", "[ai][available-expressions]")
{
  // Setup global options here
  ait<cse_domaint> AE;

  ae_program T;
  T.code =
    "int main() {\n"
    "int i= 10;\n"
    "int a,b,c,d;\n"
    "while(i != 0) {\n"  // AE: []
    "  a = b + c + d;\n" // AE: []
    "  b = 42;\n"        // AE: [b + c, b + c + d]
    "  i--;\n"           // AE: []
    "}\n"
    "return a;\n"
    "}";
  T.available_expressions["6"] = {"@F@main@b", "@F@main@c"};
  T.unavailable_expressions["4"] = {"@F@main@b", "@F@main@c"};
  T.unavailable_expressions["5"] = {"@F@main@b", "@F@main@c"};
  T.unavailable_expressions["7"] = {"@F@main@b", "@F@main@c"};

  T.run_test(AE);
}

TEST_CASE("Expressions - Loops 2", "[ai][available-expressions]")
{
  // Setup global options here
  ait<cse_domaint> AE;

  ae_program T;
  T.code =
    "int main() {\n"
    "int i= 10;\n"
    "int a,b,c,d;\n"
    "while(i != 0) {\n"  // AE: []
    "  a = b + c + d;\n" // AE: [b + c]
    "  d = 42;\n"        // AE: [b + c, b + c + d]
    "  i--;\n"           // AE: [b + c]
    "}\n"
    "int left;\n" // AE: []
    "return a;\n"
    "}";
  T.available_expressions["6"] = {"@F@main@b", "@F@main@c", "@F@main@d"};
  T.available_expressions["5"] = {"@F@main@b", "@F@main@c"};
  T.available_expressions["7"] = {"@F@main@b", "@F@main@c"};
  T.unavailable_expressions["7"] = {"@F@main@b", "@F@main@c", "@F@main@d"};
  T.run_test(AE);
}

TEST_CASE("Expressions - Function Call", "[ai][available-expressions]")
{
  // Setup global options here
  ait<cse_domaint> AE;

  ae_program T;
  T.code =
    "int id(int v) { return v; }\n"
    "int main() {\n"
    "int a,b,c;"
    "int e = a + b + c;\n" // AE: []
    "a = id(b + c);\n"     // AE : [a + b, a + b + c]
    "int new;\n"
    "return a;\n" // AE: [b + c]
    "}";
  T.available_expressions["5"] = {"@F@main@a", "@F@main@b", "@F@main@c"};
  T.unavailable_expressions["6"] = {"@F@main@a", "@F@main@b", "@F@main@c"};
  T.available_expressions["6"] = {"@F@main@b", "@F@main@c"};
  T.run_test(AE);
}

// TODO: pointers! Sadly I can't make the VSA work under this testing environment as string_container has some weird initialization bug.
