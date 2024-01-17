import json
import data_implementation
import new_main_chatbot_code

class DataImplementation:
    def __init__(self):
        pass
    
    
    def get_assistant_data(self, _request):
        self._req = _request
        _role = str(self._req.role)
        _message = str(self._req.message)
        query = _message
        chat_obj = new_main_chatbot_code.BotAssistant()
        chat_response = chat_obj.chat_assistant(query)
        # print("Chat Response in Data Implementation Class")
        # print(chat_response)
        chat_response = json.loads(chat_response)
        final_response = {
            "Question": str(query),
            "Assistant": chat_response["output"]
        }
        return final_response

