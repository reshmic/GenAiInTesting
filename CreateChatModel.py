import cohere
from cohere import Client
from cohere import client
from cohere.finetuning import FinetunedModel, Settings, BaseModel

# Create a dataset
co = cohere.Client('TNWudO7N1nJlm3LJxJXZDKm0Poltn2SaM0GGHxjt') # This is your trial API key
db_dataset = co.datasets.create(
    name="db1.0",
    type="chat-finetune-input",
    data=open("./chatdata.jsonl", "rb"),
    # eval_data=open("./db_chat_eval.jsonl", "rb")
)
#print(co.wait(php_dataset))

# Wait for dataset creation to complete
result = co.wait(db_dataset)
finetuned_model = co.finetuning.create_finetuned_model(
    request=FinetunedModel(
        name="db1.0",
        settings=Settings(
            base_model=BaseModel(base_type="BASE_TYPE_CHAT"),
            dataset_id=db_dataset.id
        )
    )
)
print(finetuned_model.finetuned_model.id, finetuned_model.finetuned_model.status)