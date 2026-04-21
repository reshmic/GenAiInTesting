import cohere
from cohere import Client
from cohere import client
from cohere.finetuning import FinetunedModel, Settings, BaseModel

# Create a dataset
co = cohere.Client('TNWudO7N1nJlm3LJxJXZDKm0Poltn2SaM0GGHxjt') # This is your trial API key
db_classify_dataset = co.datasets.create(
    name="db_classify1.0",
    type="single-label-classification-finetune-input",
    data=open("./Usecases.csv", "rb"),
    # eval_data=open("./db_chat_eval.jsonl", "rb")
)
#print(co.wait(php_dataset))

# Wait for dataset creation to complete
result = co.wait(db_classify_dataset)
finetuned_model = co.finetuning.create_finetuned_model(
    request=FinetunedModel(
        name="db_classify1.0",
        settings=Settings(
            base_model=BaseModel(base_type="BASE_TYPE_CLASSIFICATION"),
            dataset_id=db_classify_dataset.id
        )
    )
)
print(finetuned_model.finetuned_model.id, finetuned_model.finetuned_model.status)