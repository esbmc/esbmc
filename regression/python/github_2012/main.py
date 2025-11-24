from typing import List, Any

# Mock JIRA client (ESBMC can't resolve real external modules)
class JIRA:
    def __init__(self, server: str, user: str, password: str):
        self.server = server
        self.user = user
        self.password = password
        self.issues: List[str] = []

    def search_issues(self, query: str) -> List[str]:
        # Simulate search results
        if "IKUT" in query:
            return ["IKUT-123", "IKUT-456"]
        return []

    def delete_issue(self, issue: str) -> None:
        # Simulate issue deletion
        if issue in self.issues:
            self.issues.remove(issue)


# Global variables
user: str = "example_user"
password: str = "example_password"
CRTest: str = "IKUT-1126373"
jira: JIRA


def connect_to_jira() -> bool:
    global jira
    try:
        jira = JIRA(server="https://idart.mot.com", user=user, password=password)
        return True
    except Exception:
        return False


def search_issue() -> List[str]:
    global jira
    results = jira.search_issues("project=IKUT and id=" + CRTest)
    for issue in results:
        jira.delete_issue(issue)
    return results


# Simulated workflow
connected: bool = connect_to_jira()
assert connected

found: List[str] = search_issue()
print(found)
