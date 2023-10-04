import pandas as pd
from typing import Text, List, Dict, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import random

class ActionGenerateResponse(Action):
    def name(self) -> Text:
        return "action_generate_response"
    
    def load_dataset_for_category(self, category: str):
        df = pd.read_excel('./dataset.xlsx')
        filtered_data = df[df['category'] == category]
        
        dataset_for_category = [
            {"question": row['question'], "answer": row['answer']}
            for index, row in filtered_data.iterrows()
        ]
        
        return dataset_for_category

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        user_intent = tracker.latest_message["intent"]["name"]

        intent_to_category = {
            "visitor_visa_question": "Visitor Visa",
            "healthcare_question": "Healthcare",
            "permanent_residence_question": "Permanent Residence"
            # Add mappings for other intents and categories
        }

        category = intent_to_category.get(user_intent)
        if category is None:
            dispatcher.utter_message("I'm sorry, I don't have an answer for that question.")
            return []

        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = TFGPT2LMHeadModel.from_pretrained(model_name)

        dataset_for_category = self.load_dataset_for_category(category)

        response = random.choice(dataset_for_category)

        input_text = response["question"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=50, truncation=True)
        generated_output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=50256)

        response_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

        dispatcher.utter_message(response_text)

        return []

class ActionUtterAnswer(Action):
    def name(self) -> Text:
        return "utter_answer"
