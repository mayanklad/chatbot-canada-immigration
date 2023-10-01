# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from typing import Text, List, Dict, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

class ActionGenerateResponse(Action):
    def name(self) -> Text:
        return "action_generate_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the user's intent
        user_intent = tracker.latest_message["intent"]["name"]

        # Define a mapping between intents and categories in your dataset
        intent_to_category = {
            "visitor_visa_question": "Visitor Visa",
            "healthcare_question": "Healthcare",
            # Map other intents to categories
        }

        # Load the dataset for the corresponding category
        category = intent_to_category.get(user_intent)
        if category is None:
            dispatcher.utter_message("I'm sorry, I don't have an answer for that question.")
            return []

        # Load the GPT-2 model and tokenizer
        model_name = "gpt2"  # You can use a specific GPT-2 variant if needed
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # Load the dataset for the category (replace with your dataset loading logic)
        dataset_for_category = load_dataset_for_category(category)

        # Randomly select a response from the dataset
        response = random.choice(dataset_for_category)

        # Generate a response using the model
        input_text = response["question"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=50, truncation=True)
        generated_output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=50256)

        response_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

        dispatcher.utter_message(response_text)

        return []
