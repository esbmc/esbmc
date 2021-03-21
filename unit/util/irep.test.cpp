// Copyright 2018 Michael Tautschnig

/// \file Tests that irept memory consumption is fixed

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/irep.h>

SCENARIO("irept_memory", "[core][utils][irept]")
{
  GIVEN("Always")
  {
    THEN("An irept is just a pointer")
    {
      REQUIRE(sizeof(irept) == sizeof(void *));
    }

    THEN("get_nil_irep yields ID_nil")
    {
      REQUIRE(get_nil_irep().is_nil());
      REQUIRE(!get_nil_irep().is_not_nil());
    }
  }

  GIVEN("An initialized irep")
  {
    irept irep("some_id");
    irept irep_copy(irep);
    irept irep_assign = irep;

    irept irep_other("some_other_id");

    THEN("Its id is some_id")
    {
      REQUIRE(irep.id() == "some_id");
      REQUIRE(irep_copy.id() == "some_id");
      REQUIRE(irep_assign.id() == "some_id");

      REQUIRE(irep_other.id() == "some_other_id");

      // TODO(tautschnig): id_string() should be deprecated in favour of
      // id2string(...)
      REQUIRE(irep.id_string().size() == 7);
    }

    THEN("Swapping works")
    {
      irep.swap(irep_other);

      REQUIRE(irep.id() == "some_other_id");
      REQUIRE(irep_copy.id() == "some_id");
      REQUIRE(irep_assign.id() == "some_id");

      REQUIRE(irep_other.id() == "some_id");
    }
  }

  GIVEN("An irep")
  {
    irept irep;

    THEN("Its id is empty")
    {
      REQUIRE(irep.is_not_nil());
      REQUIRE(irep.id().empty());
    }

    THEN("Its id can be set")
    {
      irep.id("new_id");
      REQUIRE(irep.id() == "new_id");
    }

    THEN("find of a non-existent element yields nil")
    {
      REQUIRE(irep.find("no-such-element").is_nil());
    }

    THEN("Hashing works")
    {
      irep.id("some_id");
      irep.set("#a_comment", 42);

      REQUIRE(irep.hash() != 0);
      REQUIRE(irep.full_hash() != 0);
      REQUIRE(irep.hash() != irep.full_hash());
    }
  }

  GIVEN("Multiple ireps")
  {
    irept irep1, irep2;

    THEN("Comparison works")
    {
      REQUIRE(irep1 == irep2);
      // REQUIRE(irep1.full_eq(irep2));

      irep1.id("id1");
      irep2.id("id2");
      REQUIRE(irep1 != irep2);
      const bool one_lt_two = irep1 < irep2;
      const bool two_lt_one = irep2 < irep1;
      REQUIRE(one_lt_two != two_lt_one);
      // REQUIRE(irep1.ordering(irep2) != irep2.ordering(irep1));
      REQUIRE(irep1.compare(irep2) != 0);

      irep2.id("id1");
      REQUIRE(irep1 == irep2);
      // REQUIRE(irep1.full_eq(irep2));

      irep2.set("#a_comment", 42);
      REQUIRE(irep1 == irep2);
      // REQUIRE(!irep1.full_eq(irep2));
    }
  }
}
