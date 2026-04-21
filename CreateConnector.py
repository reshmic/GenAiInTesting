import cohere
from cohere import Client
from cohere import client
from cohere.finetuning import FinetunedModel, Settings, BaseModel

# Create a dataset
co = cohere.Client('TNWudO7N1nJlm3LJxJXZDKm0Poltn2SaM0GGHxjt') # This is your trial API key

# Create a connector using the new method
connector_name = "reqres1"
connector_url = "https://reqres.in/api/users/search" # Replace with the actual URL you want to connect to
created_connector = co.connectors.create(
        name=connector_name,
        url=connector_url,
    )
print("Created Connector:", created_connector)