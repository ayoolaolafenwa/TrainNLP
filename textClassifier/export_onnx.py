from transformer_encoder import TransformerClassifierModel
import torch
import tiktoken

def load_model(model_path):
    text_encoder = tiktoken.get_encoding("gpt2")
    num_classes = 4
    seq_len = 256
    vocab_size = text_encoder.n_vocab

    model = TransformerClassifierModel(vocab_size = vocab_size, max_len = seq_len,
    num_classes=num_classes)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
 
    model.eval()
    return model


model = load_model("news_classifier.pth")

# generate a sample input of zeros to serve as input for exporting the model to onnx
sample_input = torch.zeros(size = (1, 256)).type(torch.LongTensor)

torch.onnx.export(model, sample_input, "news_classifier.onnx", input_names=["input"], output_names=["output"], export_params=True)