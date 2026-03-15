from typing import List

class JIRA:
    def __init__(self):
        self.issues: List[str] = []

jira = JIRA()
jira.issues.append("bug1")
# This assertion should fail: the issue was appended so len should be 1, not 0
assert len(jira.issues) == 0
