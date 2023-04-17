from MultiHead import TransformerClassifierModel
import tiktoken
import torch


def load_model(model_path, vocab_size):
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



if __name__ == "__main__":
    text_encoder = tiktoken.get_encoding("gpt2")

    # Source: https://en.wikinews.org/wiki/First_deep_space_images_from_James_Webb_Space_Telescope_released
    sentence = """ 

    On Monday, NASA Administrator Bill Nelson and US President Joe Biden presented the first image (see left) obtained by the Near-Infrared Camera (NIRCam), an instrument on the James Webb Space Telescope.

    The image, named Webb's First Deep Field, shows thousands of galaxies in the SMACS 0723 galaxy cluster, about 4.6 billion light-years away from Earth. However, the area shown by the image is only a small portion of the Southern Hemisphere sky. The blended image (at left) resulted from the stitching together of numerous smaller images obtained at multiple wavelengths with far greater depth than achieved by the Hubble Space Telescope, the predecessor to James Webb.

    The telescope entered its current orbit around the L2 Lagrange point from January 24, about 1,500,000 kilometers (932,057 mi) from Earth, on the opposite side of Earth from the Sun. This followed a month-long journey that began in late December 2021, following years of delays and several cost overruns. For its expected five- to ten-year service life, it is intended to study the most distant, and therefore the earliest galaxies formed after the Big Bang.

    During its journey, dubbed "30 days of terror" by Sky & Telescope, the telescope successfully unfurled its 21 feet (6 m) wide mirror, deployed its sunshield and cooled down to below 50 degrees Kelvin as it traveled to the L2 Lagrange point.

    L2 is a secure location for spacecraft where the gravitational pull of the Sun and the Earth is balanced. Full scientific operations will involve thirteen teams of scientists. The primary mission is to find the most distant and earliest galaxies formed after the Big Bang to help study the origins of the Universe. The nominal mission time is five years, with a goal of ten. The location of its orbit is very different to the Hubble Space Telescope, which orbits much closer to Earth. James Webb's instruments face away from the Sun, giving a greater clarity to the images it will obtain compared to Hubble.

    An Ariane 5 launch vehicle carried the telescope to space on December 25 from the Guiana Space Centre in French Guiana, after arriving at the launch site in October. The launch date was delayed by a week due to unfavorable weather.

    The United States National Aeronautics and Space Administration (NASA) began project development in 1996, planning for a launch in 2007 at a cost of USD550 million. After NASA contracted Northrop Grumman to build the telescope, mission managers estimated a 2010 launch would cost between one and 3.5 billion USD. Redesigns to reduce technical requirements pushed launch plans to 2013 for an estimated cost of USD4.5 billion. The US Congress ordered a project review in 2010 which delayed the launch again to 2015.

    Due to an estimated cost of USD6.5 billion, the United States House Appropriations Subcommittee on Commerce, Justice, Science, and Related Agencies proposed canceling the telescope altogether in 2011. After a plan was made for a 2018 launch at a cost of USD8.8 billion, technical errors found in the telescope and the subsequent COVID-19 pandemic pushed the launch date to 2021.
    """ 

    text = text_encoder.encode_ordinary(sentence)
    
    seq_len = 256

    num_classes = 4

    vocab_size = text_encoder.n_vocab
    padding_value = text_encoder.eot_token


    text = torch.tensor(text, dtype=torch.long)

    if torch.cuda.is_available():
        text = text.cuda()

    text = pad_tensor(text, seq_len, padding_value)

    text = torch.unsqueeze(text, dim = 0)
    infer_model = load_model("news_classifier.pth", vocab_size=vocab_size)

    outputs = infer_model(text)

    # Classes: "world", "sports", "business", and "Science"
    classes = {0:"world", 1:"sports", 2:"business", 3:"science"}

    outputs = torch.argmax(outputs, axis = 1)
    outputs = int(outputs.cpu().numpy())
    
    result = classes[outputs]

    print(result)




