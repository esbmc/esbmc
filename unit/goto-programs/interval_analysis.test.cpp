#include <goto-programs/abstract-interpretation/interval_analysis.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <goto-programs/abstract-interpretation/interval_domain.h>

struct test_item
{
  std::string var;
  BigInt v;
  bool should_contain;
};

// TODO: maybe the map should support function_name
class test_program
{
public:
  std::string code;
  std::map<std::string, std::vector<test_item>> property;

  void run_configs(bool needs_overflow_support = false)
  {
    if (!needs_overflow_support)
    {
      // Common Interval Analysis (Linear from -infinity into +infinity)
      SECTION("Baseline")
      {
        log_status("Baseline");
        set_baseline_config();
        ait<interval_domaint> baseline;
        run_test<0>(baseline);
      }

      SECTION("Interval Arithmetic")
      {
        log_status("Interval Arithmetic");
        set_baseline_config();
        interval_domaint::enable_interval_arithmetic = true;
        ait<interval_domaint> baseline;
        run_test<0>(baseline);
      }

      SECTION("Modular Intervals")
      {
        log_status("Modular Intervals");
        set_baseline_config();
        interval_domaint::enable_modular_intervals = true;
        ait<interval_domaint> baseline;
        run_test<0>(baseline);
      }

      SECTION("Contractor")
      {
        log_status("Contractors");
        set_baseline_config();
        interval_domaint::enable_contraction_for_abstract_states = true;
        ait<interval_domaint> baseline;
        run_test<0>(baseline);
      }

      SECTION("Interval Arithmetic + Modular")
      {
        log_status("Interval Arithmetic");
        set_baseline_config();
        interval_domaint::enable_interval_arithmetic = true;
        interval_domaint::enable_modular_intervals = true;
        ait<interval_domaint> baseline;
        run_test<0>(baseline);
      }

      SECTION("Interval Arithmetic + Modular + Contractor")
      {
        log_status("Interval Arithmetic");
        set_baseline_config();
        interval_domaint::enable_interval_arithmetic = true;
        interval_domaint::enable_modular_intervals = true;
        interval_domaint::enable_contraction_for_abstract_states = true;
        ait<interval_domaint> baseline;
        run_test<0>(baseline);
      }

      // Wrapped Intervals logic (see "Interval Analysis and Machine Arithmetic 2015" paper)
      SECTION("Wrapped Intervals")
      {
        log_status("Wrapped");
        set_baseline_config();
        interval_domaint::enable_wrapped_intervals = true;
        ait<interval_domaint> baseline;
        run_test<2>(baseline);
      }
    }

    SECTION("Wrapped Intervals + Arithmetic + Bitwise")
    {
      log_status("Wrapped + Arithmetic + Bitwise");
      set_baseline_config();
      interval_domaint::enable_wrapped_intervals = true;
      interval_domaint::enable_interval_arithmetic = true;
      interval_domaint::enable_interval_bitwise_arithmetic = true;
      ait<interval_domaint> baseline;
      run_test<2>(baseline, true);
    }
  }

  static void set_baseline_config()
  {
    interval_domaint::enable_interval_arithmetic = false;
    interval_domaint::enable_interval_bitwise_arithmetic = false;
    interval_domaint::enable_modular_intervals = false;
    interval_domaint::enable_assertion_simplification = false;
    interval_domaint::enable_contraction_for_abstract_states = false;
    interval_domaint::enable_wrapped_intervals = false;

    interval_domaint::widening_under_approximate_bound = false;
    interval_domaint::widening_extrapolate = false;
    interval_domaint::widening_narrowing = false;
  }

  template <size_t Index>
  void run_test(
    ait<interval_domaint> &interval_analysis,
    bool precise_intervals = false)
  {
    auto arch_32 = goto_factory::Architecture::BIT_32;
    auto P = goto_factory::get_goto_functions(code, arch_32);
    CHECK(P.functions.function_map.size() > 0);

    interval_analysis(P.functions, P.ns);
    CHECK(P.functions.function_map.size() > 0);

    Forall_goto_functions (f_it, P.functions)
    {
      if (f_it->first == "c:@F@main")
      {
        REQUIRE(f_it->second.body_available);
        forall_goto_program_instructions (i_it, f_it->second.body)
        {
          const auto &to_check =
            property.find(i_it->location.get_line().as_string());
          if (to_check != property.end())
          {
            auto state = interval_analysis[i_it].intervals;

            for (auto property_it = to_check->second.begin();
                 property_it != to_check->second.end();
                 property_it++)
            {
              if (!property_it->should_contain && !precise_intervals)
                continue;

              const auto &value = property_it->v;

              // we need to find the actual interval however... getting the original name is hard
              auto interval_it = state->begin();
              for (; interval_it != state->end(); interval_it++)
              {
                auto real_name = interval_it->first.as_string();
                auto var_name = property_it->var;

                if (var_name.size() > real_name.size())
                  continue;

                if (std::equal(
                      var_name.rbegin(), var_name.rend(), real_name.rbegin()))
                  break;
              }

              if (interval_it == state->end())
              {
                CAPTURE(
                  precise_intervals,
                  property_it->should_contain,
                  i_it->location.get_line().as_string(),
                  value);
                REQUIRE((!precise_intervals || property_it->should_contain));
                continue; // Var not present means TOP (which is always correct)
              }

              const interval_domaint::interval &ref_variant =
                interval_it->second;
              REQUIRE(Index == ref_variant.index());
              const auto &ref = std::get<Index>(ref_variant);
              interval_domaint::integer_intervalt cpy = *ref;
              cpy.set_lower(value);
              CAPTURE(
                ref->is_top(),
                ref->is_bottom(),
                property_it->should_contain,
                cpy.get_lower());
              REQUIRE((
                ref->is_top() || ref->is_bottom() || ref->upper || ref->lower));
              ref->dump();
              REQUIRE(
                ref->contains(cpy.get_lower()) == property_it->should_contain);
            }
          }
        }
      }
    }

    log_status("Success");
  }
};

TEST_CASE("Interval Analysis - Base Sign", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;

  test_program T;
  T.code =
    "int main() {\n"
    "int a;\n"
    "int b;\n" // Here "a" should contain all values for int
    "return a;\n"
    "}";
  T.property["3"].push_back({"@F@main@a", (long)-pow(2, 31), true});
  T.property["3"].push_back({"@F@main@a", (long)pow(2, 31) - 1, true});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Base Unsigned", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;

  test_program T;
  T.code =
    "int main() {\n"
    "unsigned char a = nondet_uchar();\n"
    "int b;\n" // Here "a" should contain all values for uint
    "return a;\n"
    "}";

  T.property["3"].push_back({"@F@main@a", 0, true});
  T.property["3"].push_back({"@F@main@a", (long)pow(2, 8) - 1, true});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Ternary", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;

  test_program T;
  T.code =
    "int main() {\n"
    "char a = nondet_uchar() ? 2 : 4;\n"
    "return a;\n"
    "}";

  T.property["3"].push_back({"@F@main@a", 2, true});
  T.property["3"].push_back({"@F@main@a", 4, true});
  T.property["3"].push_back({"@F@main@a", 10, false});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Ternary (false)", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;

  test_program T;
  T.code =
    "int main() {\n"
    "char a = 0 ? 2 : 4;\n"
    "return a;\n"
    "}";

  T.property["3"].push_back({"@F@main@a", 2, false});
  T.property["3"].push_back({"@F@main@a", 4, true});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Ternary (true)", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;

  test_program T;
  T.code =
    "int main() {\n"
    "char a = 1 ? 2 : 4;\n"
    "return a;\n"
    "}";

  T.property["3"].push_back({"@F@main@a", 2, true});
  T.property["3"].push_back({"@F@main@a", 4, false});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - If/Else Statement LHS (Unsigned)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "unsigned int a = nondet_uint(); unsigned int b=0;\n "
    "if(a < 50){\n"
    "b = 2;\n" // a: [0, 50)
    "a = 52;\n"
    "b = 2;\n" // a: [52, 52]
    "} else {\n"
    "b = 4;\n" // a: [50, MAX_UINT]
    "a = 51;\n"
    "b = 4;\n" // a: [51, 51]
    "}\n"
    "return a;\n" // a : [51,52]
    "}";

  T.property["4"].push_back({"@F@main@a", 0, true});
  T.property["4"].push_back({"@F@main@a", 49, true});
  T.property["4"].push_back({"@F@main@a", 51, false});
  // TODO: ESBMC needs to convert < into <=
  //T.property["4"].push_back({"@F@main@a", 50, false});
  T.property["6"].push_back({"@F@main@a", 52, true});
  T.property["6"].push_back({"@F@main@a", 53, false});
  T.property["6"].push_back({"@F@main@a", 51, false});
  T.property["8"].push_back({"@F@main@a", 50, true});
  T.property["8"].push_back({"@F@main@a", (long long)pow(2, 31) - 1, true});
  T.property["8"].push_back({"@F@main@a", 49, false});
  T.property["8"].push_back({"@F@main@a", 0, false});
  T.property["10"].push_back({"@F@main@a", 51, true});
  T.property["12"].push_back({"@F@main@a", 51, true});
  T.property["12"].push_back({"@F@main@a", 52, true});
  T.property["12"].push_back({"@F@main@a", 53, false});
  //T.property["12"].push_back({"@F@main@a", 50, false});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - If/Else Statement LHS (Signed)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int(); int b=0;\n "
    "if(a < 50){\n"
    "b = 2;\n" // a: [MIN_INT, 50)
    "a = 52;\n"
    "b = 2;\n" // a: [52, 52]
    "} else {\n"
    "b = 4;\n" // a: [50, MAX_INT]
    "a = 51;\n"
    "b = 4;\n" // a: [51, 51]
    "}\n"
    "return a;\n" // a : [51,52]
    "}";
  T.property["4"].push_back({"@F@main@a", (long long)-pow(2, 31), true});
  T.property["4"].push_back({"@F@main@a", 49, true});
  T.property["6"].push_back({"@F@main@a", 52, true});
  T.property["8"].push_back({"@F@main@a", (long long)pow(2, 31) - 1, true});
  T.property["8"].push_back({"@F@main@a", 50, true});
  T.property["10"].push_back({"@F@main@a", 51, true});
  T.property["12"].push_back({"@F@main@a", 51, true});
  T.property["12"].push_back({"@F@main@a", 52, true});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - If/Else Statement LHS (Negative)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int(); int b=0;\n "
    "if(a <= -50){\n"
    "b = 2;\n" // a: [MIN_INT, -50)
    "a = -52;\n"
    "b = 2;\n" // a: [-52, -52]
    "} else {\n"
    "b = 4;\n" // a: [-49, MAX_INT]
    "a = 51;\n"
    "b = 4;\n" // a: [51, 51]
    "}\n"
    "return a;\n" // a : [-52,51]
    "}";

  T.property["4"].push_back({"@F@main@a", (long long)-pow(2, 31), true});
  T.property["4"].push_back({"@F@main@a", -50, true});
  T.property["6"].push_back({"@F@main@a", -52, true});
  T.property["8"].push_back({"@F@main@a", (long long)pow(2, 31) - 1, true});
  T.property["8"].push_back({"@F@main@a", -49, true});
  T.property["10"].push_back({"@F@main@a", 51, true});
  T.property["13"].push_back({"@F@main@a", -52, true});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - If/Else Statement RHS",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int(); int b=0;\n "
    "if(50 < a){\n"
    "b = 2;\n" // a: [51, MAX_INT)
    "a = 52;\n"
    "b = 2;\n" // a: [52, 52]
    "} else {\n"
    "b = 4;\n" // a: [MIN_INT, 50]
    "a = 51;\n"
    "b = 4;\n" // a: [51, 51]
    "}\n"
    "return a;\n" // a : [51,52]
    "}";

  T.property["4"].push_back({"@F@main@a", (long long)pow(2, 31) - 1, true});
  T.property["4"].push_back({"@F@main@a", 51, true});
  T.property["6"].push_back({"@F@main@a", 52, true});
  T.property["8"].push_back({"@F@main@a", (long long)-pow(2, 31), true});
  T.property["8"].push_back({"@F@main@a", 50, true});
  T.property["10"].push_back({"@F@main@a", 51, true});
  T.property["13"].push_back({"@F@main@a", 52, true});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - If/Else Statement RHS (Negative)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int(); int b=0;\n "
    "if(-50 < a){\n"
    "b = 2;\n" // a: [-49, MAX_INT)
    "a = 52;\n"
    "b = 2;\n" // a: [52, 52]
    "} else {\n"
    "b = 4;\n" // a: [MIN_INT, -50]
    "a = 51;\n"
    "b = 4;\n" // a: [51, 51]
    "}\n"
    "return a;\n" // a : [51,52]
    "}";

  T.property["4"].push_back({"@F@main@a", (long long)pow(2, 31) - 1, true});
  T.property["4"].push_back({"@F@main@a", -49, true});
  T.property["6"].push_back({"@F@main@a", 52, true});
  T.property["8"].push_back({"@F@main@a", (long long)-pow(2, 31), true});
  T.property["8"].push_back({"@F@main@a", -50, true});
  T.property["10"].push_back({"@F@main@a", 51, true});
  T.property["13"].push_back({"@F@main@a", 52, true});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - If/Else Statement LHS and RHS",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int(); int b=0; int c;\n "
    "if(a < 50){\n"
    "b = 0;\n"
    "a = 1;\n"
    "} else {\n"
    "b = 5;\n"
    "a = 10;\n"
    "}\n"
    "c = 0;\n" // a: [1,10] or [10,1] b: [0,5] or [5,0]
    "if(a < b) {\n"
    "c = 5;\n" // a: must contain 1, b: must contain 5
    "}\n"
    "return a;\n" // a : [51,52]
    "}";

  T.property["10"].push_back({"@F@main@a", 1, true});
  T.property["10"].push_back({"@F@main@a", 10, true});
  T.property["10"].push_back({"@F@main@b", 0, true});
  T.property["10"].push_back({"@F@main@b", 5, true});
  T.property["12"].push_back({"@F@main@a", 1, true});
  T.property["12"].push_back({"@F@main@b", 5, true});

  T.run_configs();
}
TEST_CASE("Interval Analysis - While Statement", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 0;\n"
    "int b = 0;\n"
    "while(a < 100) {\n" // a: [0,100]
    "b = 1;\n"           // a: [0, 99]
    "a++;\n"
    "b = 1;\n" // a: [1, 100]
    "}\n"
    "return a;\n" // a : [100, 100]
    "}";

  T.property["5"].push_back({"@F@main@a", 99, true});
  T.property["2"].push_back({"@F@main@a", 0, true});
  T.property["5"].push_back({"@F@main@a", 0, true});

  T.property["7"].push_back({"@F@main@a", 1, true});
  T.property["7"].push_back({"@F@main@a", 100, true});
  T.property["9"].push_back({"@F@main@a", 100, true});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Add Arithmetic", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 0;\n"
    "int b = 0;\n" // a: [0,0]
    "a = b + 1;\n" // a: [1,1]
    "return a;\n"
    "}";
  T.property["3"].push_back({"@F@main@a", 0, true});
  T.property["3"].push_back({"@F@main@a", 1, false});
  T.property["5"].push_back({"@F@main@a", 1, true});
  T.property["5"].push_back({"@F@main@a", 0, false});
  T.property["5"].push_back({"@F@main@a", 2, false});
  T.run_configs();
}

TEST_CASE("Interval Analysis - Add Range Arithmetic", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a;\n"
    "int b;\n"
    "if(a > 5) a = 2; else a = 4;"
    "if(b > 5) b = 1; else b = 3;\n"
    "int c = 0;\n" // a: [2,4], b: [1,3]
    "a = a + b;\n" // a: [3,7]
    "return a;\n"
    "}";
  T.property["5"].push_back({"@F@main@a", 2, true});
  T.property["5"].push_back({"@F@main@a", 4, true});
  T.property["5"].push_back({"@F@main@b", 1, true});
  T.property["5"].push_back({"@F@main@b", 3, true});
  T.property["7"].push_back({"@F@main@a", 3, true});
  T.property["7"].push_back({"@F@main@a", 7, true});
  T.property["7"].push_back({"@F@main@a", 8, false});
  T.property["7"].push_back({"@F@main@a", 2, false});
  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - Add Arithmetic (Overflow)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "char a = 127;\n"
    "char b = 0;\n" // a: [0,0]
    "a = a + 1;\n"  // a: [1,1]
    "return a;\n"
    "}";
  T.property["3"].push_back({"@F@main@a", 127, true});
  T.property["3"].push_back({"@F@main@a", -128, false});
  T.property["5"].push_back({"@F@main@a", -128, true});
  T.run_configs(true); // Needs to take overflow into account
}

TEST_CASE(
  "Interval Analysis - Add Arithmetic (Overlap)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "unsigned char a = 255;\n"
    "char b = 0;\n" // a: [255,255]
    "a = a + 1;\n"  // a: [0,0]
    "return a;\n"
    "}";
  T.property["3"].push_back({"@F@main@a", 255, true});
  T.property["3"].push_back({"@F@main@a", 0, false});
  T.property["5"].push_back({"@F@main@a", 0, true});
  T.run_configs(true); // Needs to take overflow into account
}

TEST_CASE("Interval Analysis - Sub Arithmetic", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 0;\n"
    "int b = 2;\n" // a: [0,0]
    "a = b - 1;\n" // a: [-1,-1]
    "return a;\n"
    "}";

  T.property["3"].push_back({"@F@main@a", 0, true});
  T.property["5"].push_back({"@F@main@a", 1, true});
  T.property["5"].push_back({"@F@main@a", 0, false});
  T.property["5"].push_back({"@F@main@a", 3, false});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - Sub Arithmetic (Underflow)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "char a = -128;\n"
    "char b = 1;\n" // a: [-128,-128]
    "a = a - 1;\n"  // a: [127,127]
    "return a;\n"
    "}";

  T.property["3"].push_back({"@F@main@a", -128, true});
  T.property["5"].push_back({"@F@main@a", 127, true});
  // Underflows are becoming an overaproximation
  T.property["5"].push_back({"@F@main@a", -128, false});
  T.property["5"].push_back({"@F@main@a", 126, false});

  T.run_configs(true);
}

TEST_CASE(
  "Interval Analysis - Mult Arithmetic (signed)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "char a = 10;\n"
    "if(nondet_int()) a = 1;"
    "char b = -1;\n" // b: [-1,-1]
    "a = b * a;\n"   // a: [-10,-1]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", 1, true});
  T.property["4"].push_back({"@F@main@a", 10, true});
  T.property["6"].push_back({"@F@main@a", -1, true});
  T.property["6"].push_back({"@F@main@a", -10, true});
  T.property["6"].push_back({"@F@main@a", 0, false});
  T.property["6"].push_back({"@F@main@a", -11, false});

  T.run_configs(true);
}

TEST_CASE(
  "Interval Analysis - Mult Arithmetic (signed overflow)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "char a = 100;\n" // [100, 100]
    "char b = 2;\n"   // b: [2,2]
    "a = b * a;\n"    // a: [-10,-1]
    "return a;\n"
    "}";

  T.property["6"].push_back({"@F@main@a", -56, true});
  T.run_configs(true);
}

TEST_CASE("Interval Analysis - Mult Arithmetic", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 2;\n"
    "if(nondet_int()) a = -1;"
    "int b = 8;\n" // a: [-1,2]
    "a = b * a;\n" // a: [-8,16]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", -1, true});
  T.property["4"].push_back({"@F@main@a", 2, true});
  T.property["6"].push_back({"@F@main@a", -8, true});
  T.property["6"].push_back({"@F@main@a", 16, true});

  T.run_configs(true);
}

TEST_CASE(
  "Interval Analysis - Div Arithmetic (signed)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 5;\n"
    "if(nondet_int()) a = 10;"
    "int b = -1;\n" // a: [5,10]
    "a = a / b;\n"  // a: [-5,-10]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", 5, true});
  T.property["4"].push_back({"@F@main@a", 10, true});
  T.property["6"].push_back({"@F@main@a", -5, true});
  T.property["6"].push_back({"@F@main@a", -10, true});
  T.property["6"].push_back({"@F@main@a", -11, false});
  T.property["6"].push_back({"@F@main@a", -4, false});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - Remainder (unsigned, singleton)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 10;\n"
    "int b = a % 2;\n" // [0,1]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@b", 0, true});
  T.property["4"].push_back({"@F@main@b", 1, false});

  T.run_configs(true);
}

TEST_CASE(
  "Interval Analysis - Truncation (unsigned)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "unsigned a = 250;\n"
    "if(nondet_int()) a = 260;"
    "unsigned char b = (unsigned char) a;\n"
    "return a;\n" // a: [250, 260], b: [250,5]
    "}";

  T.property["4"].push_back({"@F@main@a", 250, true});
  T.property["4"].push_back({"@F@main@a", 260, true});
  T.property["5"].push_back({"@F@main@b", 250, true});
  T.property["5"].push_back({"@F@main@b", 4, true});
  T.property["5"].push_back({"@F@main@b", 5, false});
  T.property["5"].push_back({"@F@main@b", 249, false});

  T.run_configs(true);
}

TEST_CASE("Interval Analysis - Truncation (signed)", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 250;\n"
    "if(nondet_int()) a = 260;"
    "char b = (char) a;\n"
    "return a;\n" // a: [250, 260], b: [-6,4]
    "}";

  T.property["4"].push_back({"@F@main@a", 250, true});
  T.property["4"].push_back({"@F@main@a", 260, true});
  T.property["5"].push_back({"@F@main@b", -6, true});
  T.property["5"].push_back({"@F@main@b", 4, true});
  T.property["5"].push_back({"@F@main@b", -7, false});
  T.property["5"].push_back({"@F@main@b", 5, false});

  T.run_configs(true);
}

TEST_CASE("Interval Analysis - Typecast (unsigned)", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "unsigned char a = 0;\n"
    "if(nondet_int()) a = 250;"
    "unsigned  b = (unsigned ) a;\n"
    "return a;\n" // a: [0, 250], b: [0,250]
    "}";

  T.property["4"].push_back({"@F@main@a", 0, true});
  T.property["4"].push_back({"@F@main@a", 250, true});
  T.property["5"].push_back({"@F@main@b", 0, true});
  T.property["5"].push_back({"@F@main@b", 250, true});
  T.property["5"].push_back({"@F@main@b", 251, false});

  T.run_configs(true);
}

TEST_CASE("Interval Analysis - Typecast (signed)", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "char a = -1;\n"
    "if(nondet_int()) a = -128;"
    "int  b = (int ) a;\n"
    "return a;\n" // a: [0, 250], b: [0,250]
    "}";

  T.property["4"].push_back({"@F@main@a", -1, true});   // 255
  T.property["4"].push_back({"@F@main@a", -128, true}); // 128
  T.property["4"].push_back({"@F@main@a", 127, false}); // 127
  T.property["4"].push_back({"@F@main@a", 0, false});   // 0
  T.property["5"].push_back({"@F@main@b", -1, true});
  T.property["5"].push_back({"@F@main@b", -128, true});
  T.property["5"].push_back({"@F@main@b", 128, false});
  T.property["5"].push_back({"@F@main@b", 0, false});

  T.run_configs(true);
}

TEST_CASE(
  "Interval Analysis - Typecast (unsigned to signed)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "unsigned char a = 250;\n"
    "if(nondet_int()) a = 255;"
    "char  b = (char ) a;\n"
    "return a;\n" // a: [250, 255], b: [-6,-1]
    "}";

  T.property["4"].push_back({"@F@main@a", 250, true});
  T.property["4"].push_back({"@F@main@a", 255, true});
  T.property["5"].push_back({"@F@main@b", -6, true});
  T.property["5"].push_back({"@F@main@b", -1, true});
  T.run_configs(true);
}

TEST_CASE(
  "Interval Analysis - Typecast (signed to unsigned)",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "char a = 100;\n"
    "if(nondet_int()) a = -2;"
    "unsigned char  b = (unsigned char ) a;\n"
    "return a;\n" // a: [-1, 5], b: [5,128]
    "}";

  T.property["4"].push_back({"@F@main@a", -2, true});
  T.property["4"].push_back({"@F@main@a", 100, true});
  T.property["5"].push_back({"@F@main@b", 100, true});
  T.property["5"].push_back({"@F@main@b", 254, true});
  T.property["5"].push_back({"@F@main@b", 255, false});

  T.run_configs(true);
}

TEST_CASE(
  "Interval Analysis - Typecast (unsigned to signed) 2",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "unsigned char a = -1;\n"
    "int b = a;"
    "unsigned char  c = (unsigned char ) a;\n"
    "return a;\n" // a: [-1, 5], b: [5,128]
    "}";

  T.property["4"].push_back({"@F@main@a", 255, true});
  T.property["4"].push_back({"@F@main@b", 255, true});
  T.property["4"].push_back({"@F@main@b", -1, false});

  T.run_configs(true);
}

TEST_CASE("Interval Analysis - Bitand", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 1;\n"
    "if(nondet_int()) a = 2;"
    "int b = 2;\n" // a: [0001,0010]
    "a = a & b;\n" // a: [0,0]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", 1, true});
  T.property["4"].push_back({"@F@main@a", 2, true});
  T.property["6"].push_back({"@F@main@a", 0, true});
  T.property["6"].push_back({"@F@main@a", 2, true});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Bitor", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = 1;\n"
    "if(nondet_int()) a = 2;"
    "int b = 4;\n" // a: [0001,0010]
    "a = a | b;\n" // a: [5,6]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", 1, true});
  T.property["4"].push_back({"@F@main@a", 2, true});
  T.property["6"].push_back({"@F@main@a", 5, true});
  T.property["6"].push_back({"@F@main@a", 6, true});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Left Shift", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int() ? 1 : 4;\n"
    "int b = a << 2;\n" // a: [4,16] or [16 ,4]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", 1, true});
  T.property["4"].push_back({"@F@main@a", 4, true});
  T.property["4"].push_back({"@F@main@a", 5, false});
  T.property["4"].push_back({"@F@main@a", 0, false});

  T.property["4"].push_back({"@F@main@b", 4, true});
  T.property["4"].push_back({"@F@main@b", 16, true});
  T.property["4"].push_back({"@F@main@b", 17, false});
  T.property["4"].push_back({"@F@main@b", 3, false});

  T.run_configs();
}

TEST_CASE("Interval Analysis - Right Shift", "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int() ? 10 : 20;\n"
    "int b = a >> 1;\n" // a: [5,10]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", 10, true});
  T.property["4"].push_back({"@F@main@a", 20, true});
  T.property["4"].push_back({"@F@main@a", 21, false});
  T.property["4"].push_back({"@F@main@a", 9, false});
  T.property["4"].push_back({"@F@main@b", 5, true});
  T.property["4"].push_back({"@F@main@b", 10, true});
  T.property["4"].push_back({"@F@main@b", 11, false});
  T.property["4"].push_back({"@F@main@b", 4, false});

  T.run_configs();
}

TEST_CASE(
  "Interval Analysis - Arithmetic Right Shift",
  "[ai][interval-analysis]")
{
  // Setup global options here
  ait<interval_domaint> interval_analysis;
  test_program T;
  T.code =
    "int main() {\n"
    "int a = nondet_int() ? -20 : -10;\n"
    "int b = a >> b;\n" // a: [-5,-10]
    "return a;\n"
    "}";

  T.property["4"].push_back({"@F@main@a", -10, true});
  T.property["4"].push_back({"@F@main@a", -20, true});
  T.property["4"].push_back({"@F@main@a", -21, false});
  T.property["4"].push_back({"@F@main@a", -9, false});

  T.property["4"].push_back({"@F@main@b", -5, true});
  T.property["6"].push_back({"@F@main@b", -10, true});
  T.property["6"].push_back({"@F@main@b", -11, false});
  T.property["6"].push_back({"@F@main@b", -4, false});

  T.run_configs();
}
