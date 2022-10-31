# Build A Masked Language Model with Transformers
This is a step by step guide using hugging face transformers to create a Masked Language Model to predict a masked word in a sentence.

* Note: I published a medium article on the concept of Transformers and Training with Transformers, https://olafenwaayoola.medium.com/the-concept-of-transformers-and-training-a-transformers-model-45a09ae7fb50

* Install neccessary packages
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

pip3 install transformers

pip3 install datasets

pip3 install accelerate
```

### Working with Bert Encoder Network
There are different network branches in transformers ranging from encoders like Bert, autoregressive networks like Bart to sequence to sequence networks like GPT. 
 

* Bert
It is a bidirectional transformer network that takes in corpus, generates a sequence of feature vectors for each word in a text. Bert uses its self attention mecahnism to study the contextual and semantic meaning of a sentence to predict a masked word in a sentence or classify the sentiment of a sentence in sentiment analysis. For example a sentence with a masked word like *"The [MASK] is happy."* is passed into a bert network to predict the missing masked word in the middle of the sentence. The network considers the semantic and syntactic meaning of both the left and right sides of the masked word to generate the appropriate word to fill in the blanked space. The predicted masked word must align around a living thing like person or animal, for example the full sentence  after the prediction of the masked word can be *"The student is happy."* or the *"The dog is  happy."* Understanding the context and the grammatical meaning of a sentence is very important for good result, because the model predicting the masked as an inanimate object like car or house such as *"The house is happy"* is completely wrong. In this tutorial Bert will be used to train a Masked Language Model. 

### Train A Masked Language Model ON IMDB Dataset
IMDB dataset is a popular dataset for benchmarking sentiment analysis and we shall finetune a pretrained distilbert model on the IMDB dataset to create a masked language model. 
Distilbert is a light variant of Bert. 

### Tokenize The Dataset

``` python
from datasets import load_dataset
from transformers import AutoTokenizer
#load imdb dataset
imdb_data = load_dataset("imdb")

# use bert model checkpoint tokenizer
model_checkpoint = "distilbert-base-uncased"
# word piece tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#define tokenize function to tokenize the dataset
def tokenize_function(data):
    result = tokenizer(data["text"])
    return result

# batched is set to True to activate fast multithreading!
tokenize_dataset = imdb_data.map(tokenize_function, batched = True, remove_columns = ["text", "label"])
```

Loaded the IMDB dataset and we loaded the tokenizer from the DistilBERT model, which is a word piece subword tokenizer. We created a function to tokenize the IMDb dataset.Finally we called the tokenizer function, and applied it on the loaded dataset. 

### Concatenate and Chunk Dataset

``` python
def concat_chunk_dataset(data):
    chunk_size = 128
    # concatenate texts
    concatenated_sequences = {k: sum(data[k], []) for k in data.keys()}
    #compute length of concatenated texts
    total_concat_length = len(concatenated_sequences[list(data.keys())[0]])

    # drop the last chunk if is smaller than the chunk size
    total_length = (total_concat_length // chunk_size) * chunk_size

    # split the concatenated sentences into chunks using the total length
    result = {k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_sequences.items()}

    '''we create a new labels column which is a copy of the input_ids of the processed text data,the labels column serve as 
    ground truth for our masked language model to learn from. '''
    
    result["labels"] = result["input_ids"].copy()

    return result

processed_dataset = tokenize_dataset.map(concat_chunk_dataset, batched = True)
```
In Natural Language Processing we need to set a bench mark for text sequences length to be trained, the maximum length for bert pretrained model to be used is 512. 
We concatenate all the text sequences in the dataset, set a chunk size of *128* and divide this concatenated text into chunks according to the maximum length of the pretrained model. We used a chunk size of *128* instead of *512* because of gpu utilization. If a very powerful gpu is available 512 should be used. 

### Dataset Masking 

``` python
from transformers import DataCollatorForLanguageModeling

''' Downsample the dataset to 10000 samples for training to for low gpu consumption'''
train_size = 10000

# test dataset is 10 % of the 10000 samples selected which is 1000
test_size = int(0.1 * train_size)

downsampled_dataset = processed_dataset["train"].train_test_split(train_size=train_size, test_size=test_size, seed=42)

''' Apply random masking once on the whole test data, then uses the default data collector to handle the test dataset in batches '''

data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability = 0.15)

# we shall insert mask randomly in the sentence
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


eval_dataset = downsampled_dataset["test"].map(insert_random_mask,batched=True,
remove_columns=downsampled_dataset["test"].column_names)

eval_dataset = eval_dataset.rename_columns({"masked_input_ids": "input_ids",
"masked_attention_mask": "attention_mask","masked_labels": "labels"})
```

We downsample the dataset to 10000 text samples and used 10% of the train dataset as test dataset which is 1000.  We also randomly masked the test dataset, replaced the unmasked columns in the test dataset with masked columns.


###  Prepare Training Procedure

``` python
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from transformers import default_data_collator

# set batch size to 32, a larger bacth size when using a more powerful gpu
batch_size = 32

# load the train dataset for traing
train_dataloader = DataLoader(downsampled_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator,)

# load the test dataset for evaluation
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)

# initialize pretrained bert model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# set the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# initialize accelerator for training
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

# set the number of epochs which is set to 30
num_train_epochs = 30
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# define the learning rate scheduler for training
lr_scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)
```
This part is very similar to the normal procedure used in pytorch to train a model. Pytorch *DataLoader* is used to load processed train and test datasets, pytorch optimizer to set the optimizer for training model, used transformers inbuilt model loader to load the distilbert model, transformers inbuilt learning rate scheduler to set the learning rate for training and prepare all the parameters, train and evaluation datasets with transformers accelerator for training.

### Train Dataset

``` python
import torch
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

# directory to save the models
output_dir = "MLP_TrainedModels"

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]

    # perplexity metric used for mask language model training
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
```

Finally we train the model and used *Perplexity* as the metric for evaluating the trained models. The lower the *Perplexity* the better the model.
We trained for 30 epochs and the lowest perplexity is *9.19*, you can train for longer epochs to get the lowest Perplexity possible.
The model directory is *MLP_TrainedModels* where the trained models are stored. 
**Note:** The full training code is [here](https://github.com/ayoolaolafenwa/TrainNLP/blob/main/train_masked_language_model.py)

### Test the model

``` python
from transformers import pipeline

pred_model = pipeline("fill-mask", model = "MLP_TrainedModels")

text = "This is an [MASK] movie."

preds = pred_model(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
```

The trained models are stored in *MLP_TrainedModels* and paste the directory to set the model value.
We print out a list of generated sentences from the model with the appropriate value for the masked word in the sentence. 
*Output
```
>>> this is an excellent movie.
>>> this is an amazing movie.
>>> this is an awesome movie.
>>> this is an entertaining movie.
```

I have pushed the Masked Language Model I trained to huggingface hub and it is available for testing. Check the Masked Language Model on hugging face repository. https://huggingface.co/ayoolaolafenwa/Masked-Language-Model

*Rest API Code for Testing the Masked Language Model
This is the inference API python code for testing the masked language model directly from hugging face.
``` python
import requests

API_URL = "https://api-inference.huggingface.co/models/ayoolaolafenwa/Masked-Language-Model"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Washington DC is the [MASK] of USA.",
})
print(output[0]["sequence"])
```

*Output
```
washington dc is the capital of usa.
```
It produces the correct output, *“washington dc is the capital of usa.”*

*Load the Masked Language Model with Transformers
You can easily load the Language model with transformers using this code.
``` python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("ayoolaolafenwa/Masked-Language-Model")

model = AutoModelForMaskedLM.from_pretrained("ayoolaolafenwa/Masked-Language-Model") 

inputs = tokenizer("The internet [MASK] amazing.", return_tensors="pt")


with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
output = tokenizer.decode(predicted_token_id)
print(output) 
```
```
is
```
It prints out the predicted masked word *“is”*.

*Colab Training
I created a google colab notebook with steps on creating a hugging face account, training the Masked Language Model and uploading the model to Hugging Face repository. Check the notebook https://colab.research.google.com/drive/1BymoZgVU0q02zYv1SdivK-wChG-ooMXL?usp=sharing





















