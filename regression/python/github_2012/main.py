from typing import List


class JIRA:
    def __init__(self, server: str, user: str, password: str):
        pass

    def search_issues(self, query: str) -> List[str]:
        return ["IKUT-123", "IKUT-456"]


user: str = "example_user"
password: str = "example_password"
CRTest: str = "IKUT-1126373"
jira: JIRA


def connect_to_jira() -> bool:
    global jira
    jira = JIRA(server="https://idart.mot.com", user=user, password=password)
    return True


def search_issue() -> List[str]:
    global jira
    results = jira.search_issues("project=IKUT and id=" + CRTest)
    return results


connected: bool = connect_to_jira()
assert connected
found: List[str] = search_issue()
