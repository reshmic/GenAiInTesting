import cohere
from cohere import Client
from cohere import client
from cohere.finetuning import FinetunedModel, Settings, BaseModel

# Create a dataset
co = cohere.Client('TNWudO7N1nJlm3LJxJXZDKm0Poltn2SaM0GGHxjt') # This is your trial API key

co.finetuning.delete_finetuned_model("ac713c36-9da6-48e1-b94d-f0d6b771ca31")
