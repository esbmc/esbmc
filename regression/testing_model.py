from dataclasses import dataclass, replace
import enum
from pathlib import Path
import shlex


# TestModes
# CORE -> Essential tests that are fast
# THOROUGH -> Slower tests
# KNOWNBUG -> Tests that are known to fail due to bugs
# FUTURE -> Test that are known to fail due to missing implementation
# ALL -> Run all tests
class TestMode(enum.Enum):
    CORE = "CORE"
    THOROUGH = "THOROUGH"
    KNOWNBUG = "KNOWNBUG"
    FUTURE = "FUTURE"
    ALL = "ALL"

    @staticmethod
    def from_string(mode_str: str) -> "TestMode":
        try:
            return TestMode(mode_str)
        except ValueError:
            raise ValueError(f"Unsupported test mode: {mode_str}")


FAIL_MODES: list[TestMode] = [TestMode.KNOWNBUG]


@dataclass(frozen=True)
class TestDescription:
    """Immutable test.desc fields."""

    # We use tuple for test_regex and labels to ensure immutability, as lists are mutable and would break the immutability of the dataclass.
    test_dir: Path
    # relative_dir is the path of the test directory relative to the regression root, used for generating arguments and labels
    relative_dir: Path
    test_mode: TestMode
    test_file: str
    test_args: str
    test_regex: tuple[str, ...]
    labels: tuple[str, ...]

    @staticmethod
    def parse_test_description(
        test_dir: Path, regression_root: Path
    ) -> "TestDescription":
        """Parse a test description from the given directory."""
        assert test_dir.is_absolute() and test_dir.exists(), f"Test directory does not exist or not absolute: {test_dir}"
        assert regression_root.is_absolute() and regression_root.exists(), f"Regression root does not exist or not absolute: {regression_root}"
        test_desc_path = test_dir / "test.desc"
        assert (
            test_desc_path.exists()
        ), f"Test description file does not exist: {test_desc_path}"
        with test_desc_path.open(encoding="utf-8") as fp:
            # First line - TEST MODE, Label1, Label2, ...
            test_mode_str, *test_labels = map(str.strip, fp.readline().strip().split(","))
            # assert (
            #     test_mode.strip() == test_mode
            # ), f"{test_dir}: test mode line must not have leading/trailing whitespace: '{test_mode}'"
            test_mode: TestMode = TestMode.from_string(test_mode_str)

            # Second line - test file
            test_file = fp.readline().strip()
            assert (test_dir / test_file).exists()

            # Third line - Arguments of executable
            test_args = fp.readline().strip()

            # Fourth line and beyond
            # Regex of expected output
            test_regex = tuple(line.strip() for line in fp if line.strip())
            # assert (
            #     test_regex
            # ), f"{test_dir}: at least one non-empty line of expected output regex is required"
        relative_dir = test_dir.relative_to(regression_root)
        return TestDescription(
            test_dir,
            relative_dir,
            test_mode,
            test_file,
            test_args,
            test_regex,
            tuple(test_labels),
        )

    @property
    def name(self) -> str:
        """Get the test name, which is the name of the test directory."""
        return self.test_dir.name

    def generate_run_argument_list(
        self, *tool: str, smt_only: bool = False, unsupported_options: list[str] = []
    ) -> list[str]:
        """Generates run command list to be used in Popen"""
        result: list[str] = list(tool)
        for x in shlex.split(self.test_args):
            if x != "":
                p = self.test_dir / x
                result.append(str(p) if p.exists() else x)
        if smt_only:
            result.append("--smtlib")
            result.append("--smt-formula-only")
            result.append("--output")
            result.append(f"{self.test_dir}.smt2")
            result.append("--array-flattener")

        for x in unsupported_options:
            try:
                index = result.index(x)
                result.pop(index)
                result.pop(index)
            except ValueError:
                pass

        result.append(str(self.test_dir / self.test_file))
        return result

    def save_test(self) -> None:
        """Replaces original test with the current configuration"""
        test_desc_path: Path = self.test_dir / "test.desc"
        assert (
            test_desc_path.is_file()
        ), f"Test description file does not exist: {test_desc_path}"
        with open(test_desc_path, "w") as f:
            f.write(f"{self.test_mode.value}")
            if self.labels:
                f.write(", " + ", ".join(self.labels))
            f.write("\n")
            f.write(f"{self.test_file}\n")
            f.write(f"{self.test_args}\n")
            for re in self.test_regex:
                f.write(f"{re}\n")

    def with_mode(self, new_mode: TestMode) -> "TestDescription":
        """Returns a copy of the test description with the given mode."""
        return replace(self, test_mode=new_mode)

    def with_labels(self, new_labels: list[str]) -> "TestDescription":
        """Returns a copy of the test description with the given labels."""
        return replace(self, labels=tuple(new_labels))

    def with_args(self, new_args: str) -> "TestDescription":
        """Returns a copy of the test description with the given arguments."""
        return replace(self, test_args=new_args)

    def with_regex(self, new_regex: list[str]) -> "TestDescription":
        """Returns a copy of the test description with the given expected output regex."""
        assert (
            new_regex
        ), "at least one non-empty line of expected output regex is required"
        return replace(self, test_regex=tuple(new_regex))

    def __str__(self) -> str:
        return f"[{self.name}]: {self.test_dir}, {self.test_mode}"
