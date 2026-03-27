class Agents:

    def use_inline_agent(self, outputSchema: dict) -> dict:
        response = {}
        if "properties" in outputSchema:
            for prop, spec in outputSchema["properties"].items():
                if spec["type"] == "boolean":
                    response[prop] = True
        return response


agents = Agents()

result = agents.use_inline_agent(outputSchema={
    "type": "object",
    "properties": {
        "has_price_text": {
            "type": "boolean"
        }
    }
})

assert result["has_price_text"] == True
