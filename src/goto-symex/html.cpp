#include "util/message.h"
#include <fstream>
#include <goto-symex/goto_trace.h>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include <list>
#include <util/language.h>
#include <langapi/language_util.h>

// TODO: Multiple files
// TODO: Control Trace
// TODO: Refactor this

namespace
{


} // namespace


class html_report
{
public:
  html_report(const goto_tracet &goto_trace, const namespacet &ns);
  void output(std::ostream &oss) const;

protected:
  const std::string generate_head() const;
  const std::string generate_body() const;
  const goto_tracet &goto_trace;
  std::vector<goto_trace_stept> asserts;

private:
  const namespacet &ns;
  struct code_lines
  {
    code_lines(const std::string &content) : content(content) {}
    const std::string content;
    std::string to_html() const;
  };

  struct code_steps
  {
    code_steps(const std::pair<size_t, std::string> &init) : step(init.first), msg(init.second) {}
    size_t step;
    std::string msg;
    std::string to_html(size_t last) const;
  };

  

  
  static inline const std::string tag_body_str(const std::string_view tag,const std::string_view body) {
  return fmt::format("<{0}>{1}</{0}>", tag, body);
}

  // NOTE: Scripts were shamelessly taken from clang

static constexpr std::string_view style{
  R"(<style type="text/css">
body { color:#000000; background-color:#ffffff }
body { font-family:Helvetica, sans-serif; font-size:10pt }
h1 { font-size:14pt }
.FileName { margin-top: 5px; margin-bottom: 5px; display: inline; }
.FileNav { margin-left: 5px; margin-right: 5px; display: inline; }
.FileNav a { text-decoration:none; font-size: larger; }
.divider { margin-top: 30px; margin-bottom: 30px; height: 15px; }
.divider { background-color: gray; }
.code { border-collapse:collapse; width:100%; }
.code { font-family: "Monospace", monospace; font-size:10pt }
.code { line-height: 1.2em }
.comment { color: green; font-style: oblique }
.keyword { color: blue }
.string_literal { color: red }
.directive { color: darkmagenta }

/* Macros and variables could have pop-up notes hidden by default.
  - Macro pop-up:    expansion of the macro
  - Variable pop-up: value (table) of the variable */
.macro_popup, .variable_popup { display: none; }

/* Pop-up appears on mouse-hover event. */
.macro:hover .macro_popup, .variable:hover .variable_popup {
  display: block;
  padding: 2px;
  -webkit-border-radius:5px;
  -webkit-box-shadow:1px 1px 7px #000;
  border-radius:5px;
  box-shadow:1px 1px 7px #000;
  position: absolute;
  top: -1em;
  left:10em;
  z-index: 1
}

.macro_popup {
  border: 2px solid red;
  background-color:#FFF0F0;
  font-weight: normal;
}

.variable_popup {
  border: 2px solid blue;
  background-color:#F0F0FF;
  font-weight: bold;
  font-family: Helvetica, sans-serif;
  font-size: 9pt;
}

/* Pop-up notes needs a relative position as a base where they pops up. */
.macro, .variable {
  background-color: PaleGoldenRod;
  position: relative;
}
.macro { color: DarkMagenta; }

#tooltiphint {
  position: fixed;
  width: 50em;
  margin-left: -25em;
  left: 50%;
  padding: 10px;
  border: 1px solid #b0b0b0;
  border-radius: 2px;
  box-shadow: 1px 1px 7px black;
  background-color: #c0c0c0;
  z-index: 2;
}

.num { width:2.5em; padding-right:2ex; background-color:#eeeeee }
.num { text-align:right; font-size:8pt }
.num { color:#444444 }
.line { padding-left: 1ex; border-left: 3px solid #ccc }
.line { white-space: pre }
.msg { -webkit-box-shadow:1px 1px 7px #000 }
.msg { box-shadow:1px 1px 7px #000 }
.msg { -webkit-border-radius:5px }
.msg { border-radius:5px }
.msg { font-family:Helvetica, sans-serif; font-size:8pt }
.msg { float:left }
.msg { position:relative }
.msg { padding:0.25em 1ex 0.25em 1ex }
.msg { margin-top:10px; margin-bottom:10px }
.msg { font-weight:bold }
.msg { max-width:60em; word-wrap: break-word; white-space: pre-wrap }
.msgT { padding:0x; spacing:0x }
.msgEvent { background-color:#fff8b4; color:#000000 }
.msgControl { background-color:#bbbbbb; color:#000000 }
.msgNote { background-color:#ddeeff; color:#000000 }
.mrange { background-color:#dfddf3 }
.mrange { border-bottom:1px solid #6F9DBE }
.PathIndex { font-weight: bold; padding:0px 5px; margin-right:5px; }
.PathIndex { -webkit-border-radius:8px }
.PathIndex { border-radius:8px }
.PathIndexEvent { background-color:#bfba87 }
.PathIndexControl { background-color:#8c8c8c }
.PathIndexPopUp { background-color: #879abc; }
.PathNav a { text-decoration:none; font-size: larger }
.CodeInsertionHint { font-weight: bold; background-color: #10dd10 }
.CodeRemovalHint { background-color:#de1010 }
.CodeRemovalHint { border-bottom:1px solid #6F9DBE }
.msg.selected{ background-color:orange !important; }

table.simpletable {
  padding: 5px;
  font-size:12pt;
  margin:20px;
  border-collapse: collapse; border-spacing: 0px;
}
td.rowname {
  text-align: right;
  vertical-align: top;
  font-weight: bold;
  color:#444444;
  padding-right:2ex;
}

/* Hidden text. */
input.spoilerhider + label {
  cursor: pointer;
  text-decoration: underline;
  display: block;
}
input.spoilerhider {
 display: none;
}
input.spoilerhider ~ .spoiler {
  overflow: hidden;
  margin: 10px auto 0;
  height: 0;
  opacity: 0;
}
input.spoilerhider:checked + label + .spoiler{
  height: auto;
  opacity: 1;
}
</style>)"};

static constexpr std::string_view annotated_source_header_fmt{
  R"(
<h3>Annotated Source Code</h3>
<p>Press <a href="#" onclick="toggleHelp(); return false;">'?'</a>
   to see keyboard shortcuts</p>
<input type="checkbox" class="spoilerhider" id="showinvocation" />
<label for="showinvocation" >Show analyzer invocation</label>
<div class="spoiler">{}
</div>
<div id='tooltiphint' hidden="true">
  <p>Keyboard shortcuts: </p>
  <ul>
    <li>Use 'j/k' keys for keyboard navigation</li>
    <li>Use 'Shift+S' to show/hide relevant lines</li>
    <li>Use '?' to toggle this window</li>
  </ul>
  <a href="#" onclick="toggleHelp(); return false;">Close</a>
</div>
)"};

  static constexpr std::string_view counterexample_checkbox { R"(<form>
    <input type="checkbox" name="showCounterexample" id="showCounterexample" />
    <label for="showCounterexample">
       Show only relevant lines
    </label>
</form>)"};


  static std::string counterexample_filter(const std::string relevant_lines_js)
  {
    std::ostringstream oss;
    oss << "<script type='text/javascript'>\n";
    oss << "var relevant_lines = " << relevant_lines_js << ";\n";
    oss << R"(
var filterCounterexample = function (hide) {
  var tables = document.getElementsByClassName("code");
  for (var t=0; t<tables.length; t++) {
    var table = tables[t];
    var file_id = table.getAttribute("data-fileid");
    var lines_in_fid = relevant_lines[file_id];
    if (!lines_in_fid) {
      lines_in_fid = {};
    }
    var lines = table.getElementsByClassName("codeline");
    for (var i=0; i<lines.length; i++) {
        var el = lines[i];
        var lineNo = el.getAttribute("data-linenumber");
        if (!lines_in_fid[lineNo]) {
          if (hide) {
            el.setAttribute("hidden", "");
          } else {
            el.removeAttribute("hidden");
          }
        }
    }
  }
}

window.addEventListener("keydown", function (event) {
  if (event.defaultPrevented) {
    return;
  }
  // SHIFT + S
  if (event.shiftKey && event.keyCode == 83) {
    var checked = document.getElementsByName("showCounterexample")[0].checked;
    filterCounterexample(!checked);
    document.getElementsByName("showCounterexample")[0].click();
  } else {
    return;
  }
  event.preventDefault();
}, true);

document.addEventListener("DOMContentLoaded", function() {
    document.querySelector('input[name="showCounterexample"]').onchange=
        function (event) {
      filterCounterexample(this.checked);
    };
});
</script>)";

    return oss.str();
  }

  static constexpr std::string_view arrow_scripts {R"(<script type='text/javascript'>
// Return range of numbers from a range [lower, upper).
function range(lower, upper) {
  var array = [];
  for (var i = lower; i <= upper; ++i) {
      array.push(i);
  }
  return array;
}

var getRelatedArrowIndices = function(pathId) {
  // HTML numeration of events is a bit different than it is in the path.
  // Everything is rotated one step to the right, so the last element
  // (error diagnostic) has index 0.
  if (pathId == 0) {
    // arrowIndices has at least 2 elements
    pathId = arrowIndices.length - 1;
  }

  return range(arrowIndices[pathId], arrowIndices[pathId - 1]);
}

var highlightArrowsForSelectedEvent = function() {
  const selectedNum = findNum();
  const arrowIndicesToHighlight = getRelatedArrowIndices(selectedNum);
  arrowIndicesToHighlight.forEach((index) => {
    var arrow = document.querySelector("#arrow" + index);
    if(arrow) {
      classListAdd(arrow, "selected")
    }
  });
}

var getAbsoluteBoundingRect = function(element) {
  const relative = element.getBoundingClientRect();
  return {
    left: relative.left + window.pageXOffset,
    right: relative.right + window.pageXOffset,
    top: relative.top + window.pageYOffset,
    bottom: relative.bottom + window.pageYOffset,
    height: relative.height,
    width: relative.width
  };
}

var drawArrow = function(index) {
  // This function is based on the great answer from SO:
  //   https://stackoverflow.com/a/39575674/11582326
  var start = document.querySelector("#start" + index);
  var end   = document.querySelector("#end" + index);
  var arrow = document.querySelector("#arrow" + index);

  var startRect = getAbsoluteBoundingRect(start);
  var endRect   = getAbsoluteBoundingRect(end);

  // It is an arrow from a token to itself, no need to visualize it.
  if (startRect.top == endRect.top &&
      startRect.left == endRect.left)
    return;

  // Each arrow is a very simple Bézier curve, with two nodes and
  // two handles.  So, we need to calculate four points in the window:
  //   * start node
  var posStart    = { x: 0, y: 0 };
  //   * end node
  var posEnd      = { x: 0, y: 0 };
  //   * handle for the start node
  var startHandle = { x: 0, y: 0 };
  //   * handle for the end node
  var endHandle   = { x: 0, y: 0 };
  // One can visualize it as follows:
  //
  //         start handle
  //        /
  //       X"""_.-""""X
  //         .'        \
  //        /           start node
  //       |
  //       |
  //       |      end node
  //        \    /
  //         `->X
  //        X-'
  //         \
  //          end handle
  //
  // NOTE: (0, 0) is the top left corner of the window.

  // We have 3 similar, but still different scenarios to cover:
  //
  //   1. Two tokens on different lines.
  //             -xxx
  //           /
  //           \
  //             -> xxx
  //      In this situation, we draw arrow on the left curving to the left.
  //   2. Two tokens on the same line, and the destination is on the right.
  //             ____
  //            /    \
  //           /      V
  //        xxx        xxx
  //      In this situation, we draw arrow above curving upwards.
  //   3. Two tokens on the same line, and the destination is on the left.
  //        xxx        xxx
  //           ^      /
  //            \____/
  //      In this situation, we draw arrow below curving downwards.
  const onDifferentLines = startRect.top <= endRect.top - 5 ||
    startRect.top >= endRect.top + 5;
  const leftToRight = startRect.left < endRect.left;

  // NOTE: various magic constants are chosen empirically for
  //       better positioning and look
  if (onDifferentLines) {
    // Case #1
    const topToBottom = startRect.top < endRect.top;
    posStart.x = startRect.left - 1;
    // We don't want to start it at the top left corner of the token,
    // it doesn't feel like this is where the arrow comes from.
    // For this reason, we start it in the middle of the left side
    // of the token.
    posStart.y = startRect.top + startRect.height / 2;

    // End node has arrow head and we give it a bit more space.
    posEnd.x = endRect.left - 4;
    posEnd.y = endRect.top;

    // Utility object with x and y offsets for handles.
    var curvature = {
      // We want bottom-to-top arrow to curve a bit more, so it doesn't
      // overlap much with top-to-bottom curves (much more frequent).
      x: topToBottom ? 15 : 25,
      y: Math.min((posEnd.y - posStart.y) / 3, 10)
    }

    // When destination is on the different line, we can make a
    // curvier arrow because we have space for it.
    // So, instead of using
    //
    //   startHandle.x = posStart.x - curvature.x
    //   endHandle.x   = posEnd.x - curvature.x
    //
    // We use the leftmost of these two values for both handles.
    startHandle.x = Math.min(posStart.x, posEnd.x) - curvature.x;
    endHandle.x = startHandle.x;

    // Curving downwards from the start node...
    startHandle.y = posStart.y + curvature.y;
    // ... and upwards from the end node.
    endHandle.y = posEnd.y - curvature.y;

  } else if (leftToRight) {
    // Case #2
    // Starting from the top right corner...
    posStart.x = startRect.right - 1;
    posStart.y = startRect.top;

    // ...and ending at the top left corner of the end token.
    posEnd.x = endRect.left + 1;
    posEnd.y = endRect.top - 1;

    // Utility object with x and y offsets for handles.
    var curvature = {
      x: Math.min((posEnd.x - posStart.x) / 3, 15),
      y: 5
    }

    // Curving to the right...
    startHandle.x = posStart.x + curvature.x;
    // ... and upwards from the start node.
    startHandle.y = posStart.y - curvature.y;

    // And to the left...
    endHandle.x = posEnd.x - curvature.x;
    // ... and upwards from the end node.
    endHandle.y = posEnd.y - curvature.y;

  } else {
    // Case #3
    // Starting from the bottom right corner...
    posStart.x = startRect.right;
    posStart.y = startRect.bottom;

    // ...and ending also at the bottom right corner, but of the end token.
    posEnd.x = endRect.right - 1;
    posEnd.y = endRect.bottom + 1;

    // Utility object with x and y offsets for handles.
    var curvature = {
      x: Math.min((posStart.x - posEnd.x) / 3, 15),
      y: 5
    }

    // Curving to the left...
    startHandle.x = posStart.x - curvature.x;
    // ... and downwards from the start node.
    startHandle.y = posStart.y + curvature.y;

    // And to the right...
    endHandle.x = posEnd.x + curvature.x;
    // ... and downwards from the end node.
    endHandle.y = posEnd.y + curvature.y;
  }

  // Put it all together into a path.
  // More information on the format:
  //   https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
  var pathStr = "M" + posStart.x + "," + posStart.y + " " +
    "C" + startHandle.x + "," + startHandle.y + " " +
    endHandle.x + "," + endHandle.y + " " +
    posEnd.x + "," + posEnd.y;

  arrow.setAttribute("d", pathStr);
};

var drawArrows = function() {
  const numOfArrows = document.querySelectorAll("path[id^=arrow]").length;
  for (var i = 0; i < numOfArrows; ++i) {
    drawArrow(i);
  }
}

var toggleArrows = function(event) {
  const arrows = document.querySelector("#arrows");
  if (event.target.checked) {
    arrows.setAttribute("visibility", "visible");
  } else {
    arrows.setAttribute("visibility", "hidden");
  }
}

window.addEventListener("resize", drawArrows);
document.addEventListener("DOMContentLoaded", function() {
  // Whenever we show invocation, locations change, i.e. we
  // need to redraw arrows.
  document
    .querySelector('input[id="showinvocation"]')
    .addEventListener("click", drawArrows);
  // Hiding irrelevant lines also should cause arrow rerender.
  document
    .querySelector('input[name="showCounterexample"]')
    .addEventListener("change", drawArrows);
  document
    .querySelector('input[name="showArrows"]')
    .addEventListener("change", toggleArrows);
  drawArrows();
  // Default highlighting for the last event.
  highlightArrowsForSelectedEvent();
});
</script>
  
<script type='text/javascript'>
var digitMatcher = new RegExp("[0-9]+");

var querySelectorAllArray = function(selector) {
  return Array.prototype.slice.call(
    document.querySelectorAll(selector));
}

document.addEventListener("DOMContentLoaded", function() {
    querySelectorAllArray(".PathNav > a").forEach(
        function(currentValue, currentIndex) {
            var hrefValue = currentValue.getAttribute("href");
            currentValue.onclick = function() {
                scrollTo(document.querySelector(hrefValue));
                return false;
            };
        });
});

var findNum = function() {
    var s = document.querySelector(".msg.selected");
    if (!s || s.id == "EndPath") {
        return 0;
    }
    var out = parseInt(digitMatcher.exec(s.id)[0]);
    return out;
};

var classListAdd = function(el, theClass) {
  if(!el.className.baseVal)
    el.className += " " + theClass;
  else
    el.className.baseVal += " " + theClass;
};

var classListRemove = function(el, theClass) {
  var className = (!el.className.baseVal) ?
      el.className : el.className.baseVal;
    className = className.replace(" " + theClass, "");
  if(!el.className.baseVal)
    el.className = className;
  else
    el.className.baseVal = className;
};

var scrollTo = function(el) {
    querySelectorAllArray(".selected").forEach(function(s) {
      classListRemove(s, "selected");
    });
    classListAdd(el, "selected");
    window.scrollBy(0, el.getBoundingClientRect().top -
        (window.innerHeight / 2));
    highlightArrowsForSelectedEvent();
};

var move = function(num, up, numItems) {
  if (num == 1 && up || num == numItems - 1 && !up) {
    return 0;
  } else if (num == 0 && up) {
    return numItems - 1;
  } else if (num == 0 && !up) {
    return 1 % numItems;
  }
  return up ? num - 1 : num + 1;
}

var numToId = function(num) {
  if (num == 0) {
    return document.getElementById("EndPath")
  }
  return document.getElementById("Path" + num);
};

var navigateTo = function(up) {
  var numItems = document.querySelectorAll(
      ".line > .msgEvent, .line > .msgControl").length;
  var currentSelected = findNum();
  var newSelected = move(currentSelected, up, numItems);
  var newEl = numToId(newSelected, numItems);

  // Scroll element into center.
  scrollTo(newEl);
};

window.addEventListener("keydown", function (event) {
  if (event.defaultPrevented) {
    return;
  }
  // key 'j'
  if (event.keyCode == 74) {
    navigateTo(/*up=*/false);
  // key 'k'
  } else if (event.keyCode == 75) {
    navigateTo(/*up=*/true);
  } else {
    return;
  }
  event.preventDefault();
}, true);
</script>
  
<script type='text/javascript'>

var toggleHelp = function() {
    var hint = document.querySelector("#tooltiphint");
    var attributeName = "hidden";
    if (hint.hasAttribute(attributeName)) {
      hint.removeAttribute(attributeName);
    } else {
      hint.setAttribute("hidden", "true");
    }
};
window.addEventListener("keydown", function (event) {
  if (event.defaultPrevented) {
    return;
  }
  if (event.key == "?") {
    toggleHelp();
  } else {
    return;
  }
  event.preventDefault();
});
</script>

<style type="text/css">
  svg {
      position:absolute;
      top:0;
      left:0;
      height:100%;
      width:100%;
      pointer-events: none;
      overflow: visible
  }
  .arrow {
      stroke-opacity: 0.2;
      stroke-width: 1;
      marker-end: url(#arrowhead);
  }

  .arrow.selected {
      stroke-opacity: 0.6;
      stroke-width: 2;
      marker-end: url(#arrowheadSelected);
  }

  .arrowhead {
      orient: auto;
      stroke: none;
      opacity: 0.6;
      fill: blue;
  }
</style>
<svg xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrowheadSelected" class="arrowhead" opacity="0.6"
            viewBox="0 0 10 10" refX="3" refY="5"
            markerWidth="4" markerHeight="4">
      <path d="M 0 0 L 10 5 L 0 10 z" />
    </marker>
    <marker id="arrowhead" class="arrowhead" opacity="0.2"
            viewBox="0 0 10 10" refX="3" refY="5"
            markerWidth="4" markerHeight="4">
      <path d="M 0 0 L 10 5 L 0 10 z" />
    </marker>
  </defs>
  <g id="arrows" fill="none" stroke="blue" visibility="hidden">
    <path class="arrow" id="arrow0"/>
    <path class="arrow" id="arrow1"/>
    <path class="arrow" id="arrow2"/>
    <path class="arrow" id="arrow3"/>
    <path class="arrow" id="arrow4"/>

  </g>
</svg>)"};

  
};

html_report::html_report(const goto_tracet &goto_trace, const namespacet &ns) : goto_trace(goto_trace), ns(ns)
{
  std::copy_if (goto_trace.steps.begin(), goto_trace.steps.end(), std::back_inserter(asserts), [](goto_trace_stept step){return step.is_assert();} );
  if (!asserts.size())
  {
    log_error("[HTML] Could not find any asserts in error trace!");
    abort();
  }

  if (asserts.size() > 1)
    log_warning("[HTML] Detected multiple assertions in one program trace. This is currently unsupported, only the first assertion will be dumped.");
    
}

const std::string html_report::generate_head() const
{
  std::ostringstream head;
  {
    head << tag_body_str("title", "test.c");
    head << style;
  }  
  return tag_body_str("head", head.str());
}

const std::string html_report::generate_body() const
{
  std::ostringstream body;
  // Bug Summary
  {
    const locationt &location = asserts.front().pc->location;
    const std::string filename{location.get_file().as_string()};
    const std::string position{fmt::format(
      "function {}, line {}, column {}",
      location.get_function(),
      location.get_line(),
      location.get_column())};
    std::string violation{asserts.front().comment};
    if (violation.empty())
      violation = "Assertion failure";
    violation[0] = toupper(violation[0]);
    body << "<h3>Bug Summary</h3>";
    body << R"(<table class="simpletable">)";
    body << fmt::format(
      R"(<tr><td class="rowname">File:</td><td>{}</td></tr>)", filename);
    body << fmt::format(
      R"(<tr><td class="rowname">Violation:</td><td><a href="#EndPath">{}</a><br />{}</td></tr></table>)",
      position,
      violation);
  }
  // Annoted Source Header
  {
    std::ostringstream oss;
    for (const auto &param : config.args)
      oss << param << " ";
    body << fmt::format(annotated_source_header_fmt, oss.str());
  }
  // Counter-Example filtering and Arrow drawing
  {
    std::unordered_set<size_t> relevant_lines; 
    for (const auto &step : goto_trace.steps)
    {
      if(!(step.is_assert() && step.guard))
        relevant_lines.insert(atoi(step.pc->location.get_line().c_str()));
    }

    std::ostringstream oss;
    oss << R"({"1": {)";
    for (const auto line : relevant_lines)
      oss << fmt::format( "\"{}\": {},", line, 1);
    oss << "}}";
    body << counterexample_filter(oss.str());
    body << counterexample_checkbox;
    body << arrow_scripts;
  }
  // Counter-Example and Arrows
  {

    const locationt &location = asserts.front().pc->location;
    const std::string filename{location.get_file().as_string()};
    std::vector<code_lines> lines;
    {
    std::ifstream input(filename);
    std::string line;
    while (std::getline(input, line))
      lines.push_back(line);
    }

    std::unordered_map<size_t, std::list<code_steps>> steps;
    size_t counter = 0;
    for (const auto &step : goto_trace.steps)
    {
      size_t line = atoi(step.pc->location.get_line().c_str());
      std::ostringstream oss;
      if (step.pc->is_goto())
      {
      }
      else if (step.pc->is_assume())
      {
        oss << "Assumption restriction";
      }
      else if (step.pc->is_assert())
      {
        if (step.guard)
          continue;
        std::string comment =
          (step.comment.empty() ? "Asssertion failure" : step.comment);

        comment[0] = toupper(comment[0]);
        oss << comment;
        oss << "\n" << from_expr(ns, "", step.pc->guard);
      }
      else if (step.pc->is_other())
      {
      }
      else if (step.pc->is_assign())
        {
          oss << from_expr(ns, "", step.lhs);
          if (is_nil_expr(step.value))
            oss << " (assignment removed)";
          else
            oss << " = " << from_expr(ns, "", step.value);
            
        }
        else
        {
          abort();
        }
      
      auto &list =
        steps.insert({line, std::list<code_steps>()}).first->second;

      list.push_back(std::pair{++counter, oss.str()});

      // Is this step the violation?
      if (step.is_assert() && !step.guard)
        break;
    }

    // Table begin
    body << R"(<table class="code" data-fileid="1">)";
    for (size_t i = 0; i < lines.size(); i++)
    {
      const auto &it = steps.find(i);
      if (it != steps.end())
      {
        for (const auto &step : it->second)
        {
          body << step.to_html(counter);
        }
      }
      constexpr std::string_view codeline_fmt{
        R"(<tr class="codeline" data-linenumber="{0}"><td class="num" id="LN{0}">{0}</td><td class="line">{1}</td></tr>)"};
      body << fmt::format(codeline_fmt, i+1, lines[i].to_html());
    }
    
    body << "</table>";
    // Table end
    
    
    
  }
  
  return tag_body_str("body", body.str());
}

void html_report::output(std::ostream &oss) const
{
  std::ostringstream html;  
  html << generate_head();
  html << generate_body();

  oss << "<!doctype html>";
  oss << tag_body_str("html", html.str());
  
}

void generate_html_report(
  const optionst &options,
  const std::string_view uuid,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  log_status("Generating HTML report for trace: {}", uuid);
  const html_report report(goto_trace, ns);

  std::ofstream html("report.html");
  report.output(html);
    log_status("Done");
}
#include <regex>

std::string html_report::code_lines::to_html() const
{
  constexpr std::array keywords{
    "auto",     "break",  "case",    "char",   "const",    "continue",
    "default",  "do",     "double",  "else",   "enum",     "extern",
    "float",    "for",    "goto",    "if",     "int",      "long",
    "register", "return", "short",   "signed", "sizeof",   "static",
    "struct",   "switch", "typedef", "union",  "unsigned", "void",
    "volatile", "while"};

  std::string output(content);
  for (const auto &word : keywords)
  {
    std::regex e(fmt::format("(\\b({}))([^,. ]*)", word));
    output = std::regex_replace(output,e, fmt::format("<span class='keyword'>{}</span>", word)) ;
  }
  
  //  std::string keywords_styled = std::regex_replace(content, keywords_re, "keyword!");
  return output;
}
std::string html_report::code_steps::to_html(size_t last) const
{
  constexpr double margin = 1;

  const std::string next_step =
    step + 1 == last ? "EndPath" : fmt::format("Path{}", step + 1);
  const std::string previous_step =
    step != 0 ? fmt::format("Path{}", step - 1) : "";
  
#if 0
  .PathIndex { font-weight: bold; padding:0px 5px; margin-right:5px; }
    .PathIndex { -webkit-border-radius:8px }
    .PathIndex { border-radius:8px }
    .PathIndexEvent { background-color:#bfba87 }
    .PathIndexControl { background-color:#8c8c8c }
    .PathIndexPopUp { background-color: #879abc; }
    .PathNav a { text-decoration:none; font-size: larger }
#endif


  constexpr std::string_view next_step_format{
    R"(<td><div class="PathNav"><a href="#{}" title="Next event ({}) ">&#x2192;</a></div></td>)"};


  const std::string next_step_str =
    step < last ? fmt::format(next_step_format, next_step, step + 1) : "";
  

  constexpr std::string_view previous_step_format{
    R"(<td><div class="PathNav"><a href="#{}" title="Previous event ({}) ">&#x2190;</a></div></td>)"};

  const std::string previous_step_str = step != 0 ? fmt::format(previous_step_format, previous_step, step - 1) : "";
  
  constexpr std::string_view step_format{
    R"(<tr><td class="num"></td><td class="line"><div id="Path{0}" class="msg msgEvent" style="margin-left:{1}ex"><table class="msgT"><tr><td valign="top"><div class="PathIndex PathIndexEvent">{0}</div></td>{2}<td>{3}</td>{4}</tr></table></div></td></tr>)"};

  constexpr std::string_view error_format
  {
    R"(<tr><td class="num"></td><td class="line"><div id="EndPath" class="msg msgEvent" style="margin-left:{1}ex"><table class="msgT"><tr><td valign="top"><div class="PathIndex PathIndexEvent">{0}</div></td>{2}<td>{3}</td></table></div></td></tr>)"
  };

  return fmt::format(step == last ? error_format : step_format, step,  margin*step + 1, previous_step_str, msg, next_step_str);
  
}
