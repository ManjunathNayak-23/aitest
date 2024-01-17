import sys
from fastapi import FastAPI
import uvicorn
import json
from pydantic import BaseModel
import data_implementation

app = FastAPI()

class ChatBot(BaseModel):
    role: str | None
    message: str | None

@app.get("/status")
async def status_method():
    """
    This is a test function where the application has been started from the fastapi to check the status.
    """
    return {"status": "running"}

@app.post('/bot')
async def demo(request: ChatBot):
    try:
        data_imp_obj = data_implementation.DataImplementation()
        # passing the request to data implementation class
        # response = data_imp_obj.post(request)
        response = data_imp_obj.get_assistant_data(request)
        return response
    except Exception as e:
        return {"response": str(e)}
        

if __name__ == "__main__":
    # host_url = "ai-alg-langchainchatbot-api-dev-01.azurewebsites.net"
    host_url = "localhost"
    uvicorn.run(app, host=host_url, port=8000)