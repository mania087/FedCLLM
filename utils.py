import torch
import openai
import os
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F
import time
import nltk

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def get_model_combination(local_parameters, server_model):
    combination_model_parameter = server_model.state_dict()
    for net_id, net_para in enumerate(local_parameters):
        if net_id == 0:
            for key in net_para:
                combination_model_parameter[key] = net_para[key] * (1/len(local_parameters))
        else:
            for key in net_para:
                combination_model_parameter[key] += net_para[key] * (1/len(local_parameters))
    # return the combination of local models
    return combination_model_parameter
            
def get_image_to_text_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

def predict_step(images, model, feature_extractor, tokenizer,
                 gen_kwargs={"max_length": 16, "num_beams": 4}):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def get_api_key(txt_file):
    contents = ''
    with open(txt_file) as f:
        contents = f.read()
    return contents

def get_completion(prompt, model="gpt-3.5-turbo", retry=10, waiting_time=1.0):
    messages = [{"role": "user", "content": prompt}]
    for i in range(retry):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.2, # this is the degree of randomness of the model's output
                top_p= 0.1
            )
            response_answer = response.choices[0].message["content"]
        except openai.APIError as e:
            print(f"Error: {e}")
            response_answer = e
            time.sleep(waiting_time*i) # give longer waiting time in case of time out
        else:
            break
    return response_answer

def get_data_description(data, num_sample, model, feature_extractor, tokenizer, resize=224):
    """ get data description for num_sample using the language model """
    ## set to output raw data
    data.return_raw = True
    ## get num_sample random indexes
    indexes = np.random.choice(len(data), num_sample, replace=False)
    ## transform batches into pil images
    batch_of_pil_images = [F.resize(data[x][0], resize) for x in indexes]
    ## get data description
    client_description = predict_step(batch_of_pil_images, model, feature_extractor, tokenizer)
    ## return data mode to preprocess
    data.return_raw = False

    return client_description
  
# Tokenize and lowercase the sentences
def preprocess(sentences):
    processed = [word_tokenize(sentence.lower()) for sentence in sentences]
    return processed

def sentence_embedding(sentence, model):
    words = [word for word in sentence if word in model]
    if words:
        return sum([model[word] for word in words]) / len(words)
        #return sum([model.wv[word] for word in words]) / len(words)
    else:
        return None

def normalize_value(score):
    max_val = 1
    min_val = -1
    normalized_score = (score - min_val) / (max_val - min_val)
    return normalized_score

def cosine_similarity_matrix(embeddings1, embeddings2):
    similarity_matrix = [[ normalize_value(cosine_similarity([emb1], [emb2])[0][0]) for emb2 in embeddings2] for emb1 in embeddings1]
    return similarity_matrix

def compare_sentences_score(sentence1, sentence2, model):
    list1 = preprocess(sentence1)
    list2 = preprocess(sentence2)
    
    #model = Word2Vec(list1 + list2, vector_size=100, window=5, min_count=1, sg=0)
    ## Use pretrained model
    
    # Calculate sentence embeddings for both lists
    list1_embeddings = [sentence_embedding(sentence, model) for sentence in list1]
    list2_embeddings = [sentence_embedding(sentence, model) for sentence in list2]
    
    similarity_matrix = cosine_similarity_matrix(list1_embeddings, list2_embeddings)
    
    return similarity_matrix
    

def train(net, 
          trainloader: torch.utils.data.DataLoader,
          lr: float, 
          epochs: int,
          opt: str,
          device: torch.device, 
          valloader: torch.utils.data.DataLoader = None,
          verbose=False) -> None:
    
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if opt == "adam":
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr = lr) 
    elif opt == "sgd":
        optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr = lr, momentum=0.9)
    elif opt == "rmsprop":
        optimizer = torch.optim.RMSprop([p for p in net.parameters() if p.requires_grad], lr = lr)
    
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    start_time = time.time()
    
    net.to(device)
    net.train()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_loss = 0.0
        for i, (x, y) in enumerate(trainloader):
            data, label = x.to(device), y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            
            if (i % 100 == 99) and verbose:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                
        total_loss = total_loss / len(trainloader)

    total_time = time.time() - start_time
    net.to("cpu")  # move model back to CPU
    
    # metrics
    val_loss = 0.0
    val_acc, val_f1, val_rec, val_prec = 0.0, 0.0, 0.0, 0.0

    train_loss, train_results = test(net, trainloader, device)
    if valloader:
        val_loss, test_results = test(net, valloader, device)
        val_acc = test_results["acc"]
        val_f1 = test_results["f1"]
        val_rec = test_results["rec"]
        val_prec = test_results["prec"]

    results = {
        "training_time": total_time,
        "train_loss": train_loss,
        "train_acc": train_results["acc"],
        "train_rec":train_results["rec"],
        "train_f1":train_results["f1"],
        "train_prec":train_results["prec"],
        "validation_loss": val_loss,
        "validation_acc": val_acc,
        "validation_f1":val_f1,
        "validation_rec":val_rec,
        "validation_prec": val_prec,
    }
    if verbose:
        print(f"Epoch took: {total_time:.2f} seconds")
    return results


def test(net, testloader, device: str = "cpu",  get_confusion_matrix=False):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    
    net.to(device)
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            data, labels = x.to(device), y.to(device)
            outputs = net(data)
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            # appending
            y_pred.extend(predicted.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())
    
    loss = loss / len(testloader)
    net.to("cpu")  # move model back to CPU
    
    # convert tensors to numpy arrays
    y_true = np.array(y_true,dtype=np.int64)
    y_pred = np.array(y_pred,dtype=np.int64)

    # calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    # calculate precision
    precision = precision_score(y_true, y_pred, average='macro')
    # calculate recall
    recall = recall_score(y_true, y_pred, average='macro')
    # calculate F1-score
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    results = {
        "acc":acc,
        "prec":precision,
        "rec":recall,
        "f1":f1,
    }
    if get_confusion_matrix:
        return loss, results, conf_matrix
    else:
        return loss, results