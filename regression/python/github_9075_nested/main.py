class Other:
    pass

class Schema:
    def describe(self) -> str:
        def helper(o: Schema) -> str:
            return o.__class__.__name__
        return helper(Other())

assert Schema().describe() == "Other"
