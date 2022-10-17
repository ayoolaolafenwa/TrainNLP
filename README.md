# Build A Masked Language Model with Transformers
This is a step by step guide using hugging face transformers to create a masked language model to predict a masked word in a sentence.

* Install neccessary packages
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

pip3 install transformers

pip3 install datasets

pip3 install accelerator
```

### Working with Bert Encoder Network
There are different network branches in transformers ranging from encoders like Bert, autoregressive networks like Bart or decoders to sequence to sequence networks like GPT. 
 

* Bert
It is a bidirectional transformer network that takes in corpus, generates a sequence of feature vectors for each word in a text. Bert uses its self attention mecahnism to study the contextual and semantic meaning of a sentence to predict a masked word in a sentence or classify the sentiment of a sentence in sentiment analysis. For example a sentence like *The [MASK] is happy.* is passed into a bert network to predict the missing masked word in the middle of the sentence. The network considers the semantic and syntactic meaning of both the left and right sides of the masked word to generate the appropriate word to fill in the blanked space. The predicted masked word must align around a living thing like person or animal for example the full sentence  after the prediction of the masked word can be *The student is happy.* or the *The dog is  happy.* Understanding the context and the grammatical meaning of a sentence
is very import because the model predicting the masked as an inanimate object like car or house such as *The house is happy* is completely wrong. In this tutorial Bert will be used to train a Masked Language Model. 

### Train A Masked Language Model ON IMDB Dataset
IMDB dataset is a popular dataset for benchmarking sentiment analysis and we shall finetune a pretrained bert model on the IMDB dataset to create a masked language model.

* Tokenize The Dataset

``` python

#load imdb dataset
imdb_data = load_dataset("imdb")

# use bert model checkpoint tokenizer
model_checkpoint = "bert-base-uncased"
# word piece tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#define tokenize function to tokenize the dataset
def tokenize_function(data):
    result = tokenizer(data["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

    return result

# batched is set to True to activate fast multithreading!
tokenize_dataset = imdb_data.map(tokenize_function, batched = True, remove_columns = ["text", "label"])
```

* Concatenate and Chunk Dataset

``` python
def group_texts(data):
    chunk_size = 128
    # concatenate texts
    concatenated_sequences = {k: sum(data[k], []) for k in data.keys()}
    #compute length of concatenated texts
    total_concat_length = len(concatenated_sequences[list(data.keys())[0]])

    # drop the last chunk if is smaller than the chunk size
    total_length = (total_concat_length // chunk_size) * chunk_size

    # split the concatenated sentences into chunks using the max length
    result = {k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_sequences.items()}

    '''we create a new labels column which is a copy of the input_ids of the processed text data, 
    we need to predict randomly masked tokens in the input batch and we create  a labels column to serve as 
    ground truth for our masked language model to learn from. '''
    
    result["labels"] = result["input_ids"].copy()

    return result

processed_dataset = tokenize_dataset.map(group_texts, batched = True)
```

* Dataset Masking 

``` python


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


''' we drop the unmasked columns in the test dataset and replace them with the masked ones '''

downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
```


* Prepare Training Procedure
``` python
# set batch size to 16, a larger bacth size when using a more powerful gpu
batch_size = 16

# load the train dataset for traing
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

# load the test dataset for evaluation
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

# initialize pretrained bert model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# set the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# initialize accelerator for training
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# set the number of epochs which is set to 10 
num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# define the learning rate scheduler for training
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

*   Train Dataset

``` python
progress_bar = tqdm(range(num_training_steps))

model_name = "bert-base-uncased-finetuned-imdb-accelerate"
output_dir = model_name

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















