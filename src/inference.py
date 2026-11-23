import torch 


def predict(dataset ,  tokenizer , model , batch_size=8 ) : 
    labels=[]
    confidences=[]
    model.eval()
    for start in range ( 0, len(dataset) , batch_size) :
        end =start+batch_size
        batch=dataset[start:end]
        sentence=batch["sentence"]
        tokenized_data=tokenizer(sentence , truncation=True , padding=True, return_tensors="pt")
        with torch.no_grad() : 
            output=model(**tokenized_data)
        logits=output.logits
        predictions=torch.softmax(logits , dim=-1)
        predictions=torch.max(predictions ,dim=-1)

        labels.extend(predictions.indices.cpu().tolist())
        confidences.extend(predictions.values.cpu().tolist())

    return labels , confidences