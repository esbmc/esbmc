from typing import Optional


class EchoMessageInput:

    def __init__(self, message: Optional[str]):
        self.message = message

    def _verify(self) -> None:
        if self.message is not None:
            assert 1 <= len(self.message)
            assert len(self.message) <= 10


class EchoService:

    def echo_message(self, input: EchoMessageInput) -> None:
        input._verify()


# Create client
client = EchoService()

# Create input
input_data = EchoMessageInput(message="Hello World!")

# Call the service
client.echo_message(input_data)
