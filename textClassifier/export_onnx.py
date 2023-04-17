from MultiHead import TransformerClassifierModel
import tiktoken
import torch


def load_model(model_path):
    model = TransformerClassifierModel(vocab_size = vocab_size, max_len = seq_len,
    num_classes=num_classes)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model
    
def pad_tensor(source, length, padding_value):

    new_tensor = torch.zeros(size=(length,)).to(
        dtype=source.dtype, device=source.device).fill_(torch.tensor(padding_value))

    if source.shape[0] > length:
        new_tensor[:] = source[:length]
    else:
        new_tensor[:source.shape[0]] = source

    return new_tensor


text_encoder = tiktoken.get_encoding("gpt2")

seq_len = 256

num_classes = 4

vocab_size = text_encoder.n_vocab
padding_value = text_encoder.eot_token

samplesentence = "Most companys' stocks are increasing this week." 


text = text_encoder.encode_ordinary(samplesentence)

text = torch.tensor(text, dtype=torch.long)

if torch.cuda.is_available():
    text = text.cuda()

text = pad_tensor(text, seq_len, padding_value)

text = torch.unsqueeze(text, dim = 0)
model = load_model("news_classifier.pth")
torch.onnx.export(model, text, "news_classifier.onnx", input_names=["input"], output_names=["output"], export_params=True)





