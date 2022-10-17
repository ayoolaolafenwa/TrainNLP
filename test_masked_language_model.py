from transformers import pipeline

pred_model = pipeline("fill-mask", model = "MLP_TrainedModels")

text = "This is an [MASK] movie."

preds = pred_model(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
    