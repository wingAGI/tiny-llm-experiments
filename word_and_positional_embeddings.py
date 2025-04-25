from transformers import BertModel, BertTokenizer
import numpy as np

# Load the pre-trained BERT model and tokenizer
model_path = "bert-base-multilingual-cased"             # Path to the pre-trained model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# Print the embedding shapes
word_embeddings = model.embeddings.word_embeddings
print("Word Embeddings Shape:", word_embeddings.weight.shape)           

position_embeddings = model.embeddings.position_embeddings
print("Positional Encoding Shape:", position_embeddings.weight.shape)  

# Calculate the cosine similarity between word and position embeddings
res_ww, res_pp, res_wp = [], [], []
for _ in range(100):
    i, j = np.random.randint(0, 500), np.random.randint(0, 500)         # choose two random indices randomly
    word_embedding_i = model.embeddings.word_embeddings.weight[i].detach().numpy()          
    word_embedding_j = model.embeddings.word_embeddings.weight[j].detach().numpy()
    postion_embedding_i = model.embeddings.position_embeddings.weight[i].detach().numpy()  
    position_embedding_j = model.embeddings.position_embeddings.weight[j].detach().numpy()  

    def cos(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    res_ww.append(abs(cos(word_embedding_i, word_embedding_j)))
    res_pp.append(abs(cos(postion_embedding_i, position_embedding_j)))
    res_wp.append(abs(cos(word_embedding_i, position_embedding_j)))
    
print("Word-Word Cosine Similarity, mean:", np.mean(res_ww), 'std:', np.std(res_ww))
print("Position-Position Cosine Similarity, mean:", np.mean(res_pp), 'std:', np.std(res_pp))
print("Word-Position Cosine Similarity, mean:", np.mean(res_wp), 'std:', np.std(res_wp))   