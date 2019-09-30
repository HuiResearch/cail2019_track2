import json

def getSentences(filename):
    sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            temp = []
            documents = json.loads(line)
            for content in documents:
                sentence = content["sentence"]
                temp.append(sentence)
            sentences.append(temp)
    return sentences

def work(task):
    small = "data/" + task + "/data_small_selected.json"
    large = "data/" + task + "/train_selected.json"
    oufname = "data/" + task + "/pretrain.txt"
    all_sentences = []
    small_sentences = getSentences(small)
    large_sentences = getSentences(large)
    all_sentences.extend(small_sentences)
    all_sentences.extend(large_sentences)
    ouf = open(oufname, "w", encoding="utf-8")
    for document in all_sentences:
        for sentence in document:
            ouf.write(str(sentence) + "\n")
        ouf.write("\n")
    ouf.close()

if __name__ == '__main__':
    tasks = ["divorce", "labor", "loan"]
    for task in tasks:
        work(task)
