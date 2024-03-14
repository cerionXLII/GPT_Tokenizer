#This is a class representation of the Byte Pair Encoding Tokenizer


class bpe_tokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab
        self.merges = merges


    #Go through each pair of integers and count how many times they appear
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            if pair in counts:
                counts[pair] += 1
            else:
                counts[pair] = 1

        return counts
    

    #Take a list of ids, and a tuple pair, then search and replace that pair with the new idx
    def merge_tokens(self, tokens, pair, new_token):
        new_ids = []
        i = 0
        #Check if the pair is in the list of tokens if not just return the tokens
        if pair[0] not in tokens or pair[1] not in tokens:
            return tokens

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                #We should replase the pair with the new token
                new_ids.append(new_token)
                i += 2
            else:
                #We should keep the token as is
                new_ids.append(tokens[i])
                i += 1
        return new_ids


    def train(self, text, max_merges):
        merges = {}

        utfBytes = text.encode('utf-8')

        #Map bytes to integers, now our starting point as tokens
        tokens = list(utfBytes)
        idx = 256 #Start the last known token in our vocabulary
        ids = list(tokens) #Copy the list of tokens so that we don't modify the original list
        for i in range(max_merges):
            stats = self.get_stats(ids)
            top_pair = max(stats, key=stats.get) #Get the top pair

            occurances = stats[top_pair]
            if occurances < 2:
                print(f'Occurances is less than 2, breaking at {i}')
                break
        
            idx += 1
            print(f'Merging: {top_pair} to the new token: {idx}')
            ids = self.merge_tokens(ids, top_pair, idx)

            #Save it to the merges dictionary
            merges[top_pair] = idx

        #Save the merges
        self.merges = merges

        vocab = {idx: bytes([idx]) for idx in range(256)} #Start with the original bytes as tokens, a byte is 8 bits, which is 256 possible values
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1] #The new token is the concatenation of the two previously known tokens

        self.vocab = vocab

    def encode(self, text):
        #Merges must be in the correct order, starting from the first pair to the last
        tokens = list(text.encode('utf-8')) #Bytes --> list of integers

        #Go through all the merges and replace the tokens with the more complex tokens
        for pair, id in self.merges.items():
            tokens = self.merge_tokens(tokens, pair, id)
        
        return tokens
    
    def decode(self, ids):

        byteArray = [self.vocab[i] for i in ids ] #Transform each integer to its corresponding string in the vocab, if not found, use the unknown token
        text = b''.join(byteArray) #then concatinate them
        text = text.decode('utf-8', errors='replace') #Decode the bytes to a string, if there are bytes that are not valid UTF-8, replace them with the unknown token
        return text