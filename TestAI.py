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

    ai = AIFunc.AI(mode=AIFunc.Mode.EMOTIONCLASS , model_path="./models/class_emotion_ai.pkl")
    labels = ai.emotions_label()
    # AIFunc.Train_Emotion_AI()
    print([ai.predict_message(tweet) for tweet in tweetsText])
    

    print("end")

