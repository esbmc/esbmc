from typing import List


class JIRA:

    def __init__(self):
        self.issues: List[str] = []

    def delete(self, issue: str):
        if issue in self.issues:
            self.issues.remove(issue)


jira = JIRA()
jira.issues.append("a")
jira.delete("a")
assert len(jira.issues) == 0
