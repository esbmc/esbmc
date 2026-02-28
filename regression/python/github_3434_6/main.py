class Log:

    def info(self, msg: str) -> None:
        print(msg)


class Agents:

    def use_inline_agent(self, has_boolean: bool) -> dict:
        response = {}
        if has_boolean:
            response["has_price_text"] = True
        return response


agents = Agents()
log = Log()
result = agents.use_inline_agent(has_boolean=True)
log.info(msg=result["has_price_text"])  # Passing boolean where string expected
