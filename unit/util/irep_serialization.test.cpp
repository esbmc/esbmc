#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <sstream>
#include <util/irep/irep_serialization.h>

namespace
{
irept round_trip(const irept &src)
{
  irep_serializationt::ireps_containert write_container;
  irep_serializationt writer(write_container);
  std::ostringstream out;
  writer.reference_convert(src, out);

  irep_serializationt::ireps_containert read_container;
  irep_serializationt reader(read_container);
  std::istringstream in(out.str());
  irept dst;
  reader.reference_convert(in, dst);
  return dst;
}
} // namespace

SCENARIO(
  "goto-binary ireps survive a write/read round trip",
  "[core][utils][irep_serialization]")
{
  GIVEN("a tree whose children are numbered after their parent")
  {
    irept root("root");
    root.get_sub().push_back(irept("first_child"));
    root.get_sub().push_back(irept("second_child"));

    THEN("every child reads back in its own position")
    {
      irept out = round_trip(root);
      REQUIRE(full_eq(out, root));
      REQUIRE(out.get_sub().size() == 2);
      REQUIRE(out.get_sub()[0].id() == "first_child");
      REQUIRE(out.get_sub()[1].id() == "second_child");
    }
  }

  GIVEN("a subtree that occurs twice under one parent")
  {
    irept shared("shared");
    shared.get_sub().push_back(irept("leaf"));

    irept root("root");
    root.get_sub().push_back(shared);
    root.get_sub().push_back(irept("between"));
    root.get_sub().push_back(shared);

    THEN("the back reference resolves to the first occurrence")
    {
      irept out = round_trip(root);
      REQUIRE(full_eq(out, root));
      REQUIRE(out.get_sub().size() == 3);
      REQUIRE(full_eq(out.get_sub()[0], out.get_sub()[2]));
      REQUIRE(out.get_sub()[0].get_sub()[0].id() == "leaf");
      REQUIRE(out.get_sub()[1].id() == "between");
    }
  }

  GIVEN("named subs and comments alongside subs")
  {
    irept root("root");
    root.get_sub().push_back(irept("positional"));
    root.add("named").id("named_value");

    THEN("each slot reads back with its own content")
    {
      irept out = round_trip(root);
      REQUIRE(full_eq(out, root));
      REQUIRE(out.get_sub()[0].id() == "positional");
      REQUIRE(out.find("named").id() == "named_value");
    }
  }
}
