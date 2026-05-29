import os
import re
from benchexec.tools.sv_benchmarks_util import (
    get_data_model_from_task,
    get_single_non_witness_input_file,
    ILP32,
    LP64,
)
import benchexec.tools.template
import benchexec.result as result
import decimal


class Tool(benchexec.tools.template.BaseTool2):
    """
    Tool adaptor for ESBMC (witness-aware variant).
    Extends the upstream adapter to support tasks with multiple input files,
    e.g. ViolationWitnesses / CorrectnessWitnesses task sets where the witness
    file is listed alongside the C program in input_files.
    """

    def executable(self, tool_locator):
        return tool_locator.find_executable("esbmc-wrapper.py")

    def working_directory(self, executable):
        return os.path.dirname(executable)

    def version(self, executable):
        return self._version_from_tool(executable, "-v")

    def name(self):
        return "ESBMC"

    def project_url(self):
        return "http://www.esbmc.org/"

    def cmdline(self, executable, options, task, rlimits):
        data_model_param = get_data_model_from_task(task, {ILP32: "32", LP64: "64"})
        if data_model_param and "--arch" not in options:
            options += ["--arch", data_model_param]
        return (
            [executable]
            + ["-p", task.property_file]
            + options
            + [get_single_non_witness_input_file(task)]
        )

    def determine_result(self, run):
        status = result.RESULT_UNKNOWN

        if run.output.any_line_contains("FALSE_DEREF"):
            status = result.RESULT_FALSE_DEREF
        elif run.output.any_line_contains("FALSE_FREE"):
            status = result.RESULT_FALSE_FREE
        elif run.output.any_line_contains("FALSE_MEMTRACK"):
            status = result.RESULT_FALSE_MEMTRACK
        elif run.output.any_line_contains("FALSE_MEMCLEANUP"):
            status = result.RESULT_FALSE_MEMCLEANUP
        elif run.output.any_line_contains("FALSE_OVERFLOW"):
            status = result.RESULT_FALSE_OVERFLOW
        elif run.output.any_line_contains("FALSE_TERMINATION"):
            status = result.RESULT_FALSE_TERMINATION
        elif run.output.any_line_contains("FALSE_DATARACE"):
            status = result.RESULT_FALSE_DATARACE
        elif run.output.any_line_contains("FALSE"):
            status = result.RESULT_FALSE_REACH
        elif run.output.any_line_contains("TRUE"):
            status = result.RESULT_TRUE_PROP
        elif run.output.any_line_contains("DONE"):
            status = result.RESULT_DONE

        if status == result.RESULT_UNKNOWN:
            if run.was_timeout:
                status = result.RESULT_TIMEOUT
            elif not run.output.any_line_contains("Unknown"):
                status = "ERROR"

        return status

    def get_value_from_output(self, output, identifier):
        regex = re.compile(identifier)
        matches = []
        for line in output:
            match = regex.search(line.strip())
            if match and len(match.groups()) >= 1:
                matches.append(match.group(1))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return sum(decimal.Decimal(value) for value in matches)
        return None
