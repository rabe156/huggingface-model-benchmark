from transformers import pipeline , AutoTokenizer , AutoModelForSequenceClassification

def load_model(model_name):
    Tokenizer=AutoTokenizer.from_pretrained(model_name)
    try:
        Classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
    except OSError as error:
        raise RuntimeError(
            f"Could not load model: {model_name}"
        ) from error
    
    Classifier.eval()
    return Tokenizer ,Classifier ,Classifier.config.id2label

