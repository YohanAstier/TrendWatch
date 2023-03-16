import json
import AI.AI as AIFunc

if __name__ == "__main__":
    # Opening JSON file
    print("starting...")
    f = open('musk.json')
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    tweetsText = [message['text'] for message in data['data'] ]
    # Iterating through the json

    # AIFunc.Train_Emotion_AI()
    print([AIFunc.load_and_test(tweet, "./models/class_emotion_ai.pkl", AIFunc.Mode.EMOTIONCLASS) for tweet in tweetsText])

    print("end")
