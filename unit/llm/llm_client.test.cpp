/*******************************************************************\
Module: Unit tests for src/llm
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <llm/llm_client.h>

TEST_CASE("stub backend echoes fixed model and message count", "[llm]")
{
  llm::configt cfg;
  cfg.backend = llm::backendt::stub;
  cfg.model = "unit-test-model";

  auto client = llm::make_client(cfg);
  REQUIRE(client != nullptr);
  REQUIRE(client->config().model == "unit-test-model");

  std::vector<llm::message_t> msgs = {
    {"system", "You are helpful."}, {"user", "Hello."}};

  std::string got = client->complete(msgs);
  REQUIRE(got == "[unit-test-model] got 2 message(s)");
}

TEST_CASE("cli backend returns the process output", "[llm]")
{
  llm::configt cfg;
  cfg.backend = llm::backendt::cli;
  cfg.executable = "cat";

  auto client = llm::make_client(cfg);
  REQUIRE(
    client->complete({{"user", "Hello."}}) == "user: Hello.\nassistant: ");
}

TEST_CASE("factory picks backend by enum", "[llm]")
{
  llm::configt cfg;
  cfg.backend = llm::backendt::cli;
  cfg.executable = "/bin/echo";
  cfg.model = "cli-model";

  auto client = llm::make_client(cfg);
  REQUIRE(client != nullptr);
  REQUIRE(client->config().backend == llm::backendt::cli);
}
