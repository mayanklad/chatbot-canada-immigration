import re
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import torch
class CustomChatbotAction(Action):
    def name(self) -> Text:
        return "action_custom_chatbot"

    def __init__(self):
        # self.MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', 'llm_model.pt'))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('mayanklad/faq-canada-immigration-tokenizer')
        # self.tokenizer.add_special_tokens({"pad_token": "<pad>",
        #                         "bos_token": "<startofstring>",
        #                         "eos_token": "<endofstring>"})
        # self.tokenizer.add_tokens(["<bot>:"])
        
        self.model = GPT2LMHeadModel.from_pretrained('mayanklad/faq-canada-immigration')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)

    def generate_response(self, user_input):
        user_input = "<startofstring> " + user_input + " <bot>: "
        user_input = self.tokenizer(user_input, return_tensors="pt")
        
        X = user_input["input_ids"].to(self.device)
        a = user_input["attention_mask"].to(self.device)

        output = self.model.generate(X, attention_mask=a, max_length=50)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        output = re.split('<bot>:', output)[-1].strip()
        output = re.split("<end", output, 1)[0].strip()
        
        return output

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get('text')
        nlu_metrics = tracker.latest_message

        # Generate response using custom GPT model
        response = self.generate_response(user_input)

        dispatcher.utter_message(text=response, json_message=nlu_metrics)

        return []
