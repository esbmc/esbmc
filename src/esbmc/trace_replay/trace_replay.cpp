#include <esbmc/trace_replay/trace_replay.h>
#include <goto-symex/trace/goto_trace.h>
#include <langapi/language_util.h>
#include <util/base/filesystem.h>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <charconv>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <unistd.h>
#else
#  include <io.h>
#endif

namespace
{
struct replay_eventt
{
  unsigned step;
  unsigned line = 0;
  std::string file;
  std::string function;
  std::string description;
  std::string variable_id;
  std::string variable_name;
  std::string value;
  bool failure = false;
};

std::string terminal_text(std::string text)
{
  for (char &c : text)
  {
    if (static_cast<unsigned char>(c) < 32 || c == 127)
      c = ' ';
  }
  return text;
}

std::vector<replay_eventt>
replay_events(const namespacet &ns, const goto_tracet &trace, bool proof_model)
{
  std::vector<replay_eventt> events;
  for (const auto &step : trace.steps)
  {
    const auto &location = step.pc->location;
    const std::string file = location.get_file().as_string();
    const bool failure = step.is_assert() && !step.guard;
    if (
      !failure && (file.empty() || file == "esbmc_intrinsics.h" ||
                   file_operations::is_bundled_source(file)))
      continue;

    replay_eventt event;
    event.step = step.step_nr;
    event.file = file;
    event.function = location.get_function().as_string();
    const std::string line = location.get_line().as_string();
    std::from_chars(line.data(), line.data() + line.size(), event.line);
    event.failure = failure;

    if (step.is_assignment())
    {
      event.description = "Assignment";
      if (
        !is_nil_expr(step.lhs) && is_symbol2t(step.lhs) &&
        is_number_type(step.lhs))
      {
        event.variable_id = to_symbol2t(step.lhs).thename.as_string();
        event.variable_name = from_expr(ns, "", step.lhs);
        event.value = is_nil_expr(step.value) || step.uninitialized_assignment
                        ? "unknown"
                        : from_expr(ns, "", step.value);
        event.description += ": " + event.variable_name + " = " + event.value;
      }
      else
        event.description += " (non-scalar state is not displayed)";
    }
    else if (failure)
      event.description =
        (proof_model ? "Unproven property: " : "Failure: ") + step.comment;
    else if (step.is_branching())
      event.description = "Branch: " + from_expr(ns, "", step.pc->guard);
    else
      continue;

    events.push_back(std::move(event));
    if (failure)
      break;
  }
  return events;
}

using source_cachet = std::map<std::string, std::vector<std::string>>;

const std::vector<std::string> &
source_lines(source_cachet &cache, const std::string &file)
{
  const auto inserted = cache.emplace(file, std::vector<std::string>{});
  auto &lines = inserted.first->second;
  if (inserted.second)
  {
    const auto contents = file_operations::filesystemt::get().read(file);
    if (contents)
    {
      std::istringstream input(std::string{contents->view()});
      std::string line;
      while (std::getline(input, line))
        lines.push_back(terminal_text(std::move(line)));
    }
  }
  return lines;
}

ftxui::Element source_view(source_cachet &cache, const replay_eventt &event)
{
  using namespace ftxui;
  Elements rows;
  const auto &lines = source_lines(cache, event.file);
  if (event.line == 0 || event.line > lines.size())
    rows.push_back(text("Source unavailable at this location"));
  else
  {
    const size_t current = event.line - 1;
    const size_t start = current > 5 ? current - 5 : 0;
    const size_t end = std::min(lines.size(), current + 6);
    for (size_t i = start; i < end; ++i)
    {
      auto row = text(
        (i == current ? "> " : "  ") + std::to_string(i + 1) + "  " + lines[i]);
      if (i == current)
        row = row | inverted;
      rows.push_back(row);
    }
  }
  return window(text(terminal_text(event.file)), vbox(rows)) | flex;
}

ftxui::Element
variable_view(const std::vector<replay_eventt> &events, size_t selected)
{
  using namespace ftxui;
  std::map<std::string, std::pair<std::string, std::string>> variables;
  for (size_t i = 0; i <= selected; ++i)
  {
    const auto &event = events[i];
    if (!event.variable_id.empty())
      variables[event.variable_id] = {event.variable_name, event.value};
  }

  Elements rows;
  for (const auto &entry : variables)
    rows.push_back(
      text(terminal_text(entry.second.first + " = " + entry.second.second)));
  if (rows.empty())
    rows.push_back(text("No recorded scalar values"));
  return window(text("Recorded variables"), vbox(rows)) | flex;
}
} // namespace

void replay_goto_trace(
  const namespacet &ns,
  const goto_tracet &trace,
  const std::string &context,
  bool proof_model)
{
#ifndef _WIN32
  const bool interactive = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
#else
  const bool interactive = _isatty(_fileno(stdin)) && _isatty(_fileno(stdout));
#endif
  if (!interactive)
  {
    log_status(
      "Trace replay ({}): requires an interactive terminal; using text output",
      context);
    return;
  }
  if (std::any_of(trace.steps.begin(), trace.steps.end(), [](const auto &step) {
        return step.thread_nr != 0;
      }))
  {
    log_status("Trace replay currently supports single-threaded traces only");
    return;
  }

  const auto events = replay_events(ns, trace, proof_model);
  if (events.empty())
  {
    log_status("No source events available for trace replay");
    return;
  }

  using namespace ftxui;
  size_t selected = 0;
  source_cachet sources;
  auto screen = ScreenInteractive::Fullscreen();
  screen.TrackMouse(false);
  auto renderer = Renderer([&] {
    const auto &event = events[selected];
    return vbox(
      {text("ESBMC trace replay: " + terminal_text(context)) | bold,
       hbox({source_view(sources, event), variable_view(events, selected)}) |
         flex,
       text(
         "Event " + std::to_string(selected + 1) + "/" +
         std::to_string(events.size()) + "  Trace step " +
         std::to_string(event.step) + "  " + terminal_text(event.function)),
       text(terminal_text(event.description)),
       text(
         proof_model
           ? "Proof-phase model; not a confirmed program counterexample."
           : "Values are recorded after the selected event; trace may omit "
             "operations."),
       text("[n/Right] Next  [p/Left] Previous  [e] Property  [q] Return")});
  });
  auto component = CatchEvent(renderer, [&](Event event) {
    if (event == Event::Character('q'))
      screen.ExitLoopClosure()();
    else if (event == Event::Character('n') || event == Event::ArrowRight)
      selected = std::min(selected + 1, events.size() - 1);
    else if (event == Event::Character('p') || event == Event::ArrowLeft)
      selected -= selected > 0;
    else if (event == Event::Character('e'))
    {
      const auto failure = std::find_if(
        events.begin(), events.end(), [](const auto &e) { return e.failure; });
      if (failure != events.end())
        selected = static_cast<size_t>(failure - events.begin());
    }
    else
      return false;
    return true;
  });

  std::cout.flush();
  std::cerr.flush();
  screen.Loop(component);
}
