import onnxruntime as ort
import tiktoken
import numpy as np

def load_onnxModel(model_path):
    session = ort.InferenceSession(model_path)
    return session

    
def pad_tensor(source, length, padding_value):

    new_tensor = np.zeros(shape=(length,))
    new_tensor.fill(padding_value)
    new_tensor = new_tensor.astype(np.int64)
    
    if source.shape[0] > length:
        new_tensor[:] = source[:length]
    else:
        new_tensor[:source.shape[0]] = source

    return new_tensor

if __name__ == "__main__":
    text_encoder = tiktoken.get_encoding("gpt2")
    seq_len = 256

    padding_value = text_encoder.eot_token

    # Source: https://en.wikinews.org/wiki/Coolum_win_tier_1_women%E2%80%99s_cricket_premiership_on_Australia%E2%80%99s_Sunshine_Coast
    sentence = """ Coolum claimed the premiership in the top tier of women's cricket on Australia's Sunshine Coast Sunday, defeating Hinterland by 83 runs in the Grand Final.

    Batting first, Coolum scored 181 runs for the loss of 3 wickets off their allotted 30 overs.

    3:26
    Interview with Coolum captain Sammy Franks.
    Audio: Patrick Gillett.
    3:08
    Interview with Hinterland captain and SCCA Women's committee chair Mel Shelley.
    Audio: Patrick Gillett.
    Batting after Coolum, Hinterland scored 98 runs for the loss of 8 wickets.

    Coolum captain Sammy Franks said: "The girls played so well today. 181 with the bat, that's pretty solid [...] We knew we had to put a lot of runs on the board. Aiming for 140, 150. But yeah. 181, that was wonderful."

    Two senior Coolum players, Kerry Cowling and Paula McKie, retired after the match.

    Franks continued, "It was great to win. Give Kezza and Paula a final farewell. Yeah, Paula is retiring as well. It was a little bit of a secret. She's moving on to bigger and better things. We'll miss her. She was obviously vice captain, and cricket brain and always there for me.

    "Kerry's done so much for the club. All the little things that didn't get seen. Cleaning the toilets, bringing the sausages, just organising the whole team, all of the admin stuff and just being there like a cricket mum to us," she said. """



    text = text_encoder.encode_ordinary(sentence)

    text = np.array(text).astype(np.long)


    text = pad_tensor(text, seq_len, padding_value)

    text = np.expand_dims(text, axis=0)

    onnx_model = load_onnxModel("news_classifier.onnx")
    outputs = onnx_model.run(None, {"input": text})

    outputs = np.argmax(outputs[0], axis = 1)
    
    outputs = int(outputs)

    # Classes: "world", "sports", "business", and "Science"
    classes = {0:"world", 1:"sports", 2:"business", 3:"science"}
    result = classes[outputs]

    print(result)